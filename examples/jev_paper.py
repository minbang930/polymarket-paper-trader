"""Jev + Polymarket paper-trading experiment.

Jev is reached through OpenRouter's Decisions API. It never sends real orders:
all execution is delegated to pm_trader.Engine, which reads live Polymarket books
and records simulated fills in the local SQLite paper account.

Quick start (PowerShell):
    $env:OPENROUTER_API_KEY="sk-or-..."
    python examples/jev_paper.py --cycles 20

Use --cycles 0 to run until Ctrl+C.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from pm_trader.engine import Engine
from pm_trader.models import Market, NotInitializedError, OrderBook, SimError


OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_JEV_MODEL = "~typesafe/jev-latest"

QUESTIONS: dict[str, dict[str, Any]] = {
    "direction": {
        "type": "choice",
        "instructions": (
            "For this prediction market, classify the stronger short-horizon "
            "price pressure from the supplied live order-book state. Choose neutral "
            "when the evidence is mixed or weak."
        ),
        "criteria": {
            "yes": "YES has the stronger short-horizon upward pressure.",
            "no": "NO has the stronger short-horizon upward pressure.",
            "neutral": "There is no sufficiently clear directional pressure.",
        },
    },
    "intent": {
        "type": "choice",
        "instructions": (
            "Given the current PAPER position and market state, classify the next "
            "paper-trading intent. Avoid unnecessary churn."
        ),
        "criteria": {
            "enter": "A new paper position is justified now.",
            "hold": "Do not change the paper position now.",
            "exit": "Reduce the current paper position now.",
        },
    },
    "regime": {
        "type": "choice",
        "instructions": (
            "Classify the current market microstructure regime. Treat abrupt "
            "imbalance, thin depth, or unstable/wide spreads as more dangerous."
        ),
        "criteria": {
            "calm": "Stable enough for normal paper decisions.",
            "trending": "Directional movement exists but the book remains usable.",
            "toxic": "Conditions are unstable enough that new entries should stop.",
        },
    },
    "quality": {
        "type": "choice",
        "instructions": (
            "Rate how coherent the supplied evidence is for a directional paper "
            "trade. Penalize conflicting signals, thin books, and wide spreads."
        ),
        "criteria": {
            "poor": "Evidence is weak or conflicting.",
            "weak": "Some evidence exists but is not compelling.",
            "fair": "Evidence is coherent enough to consider after code filters.",
            "strong": "Evidence is unusually coherent for this state.",
        },
    },
}


class JevError(RuntimeError):
    """OpenRouter/Jev request or response error."""


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]

    @property
    def confidence(self) -> float:
        return float(self.probabilities.get(self.choice, 0.0))


@dataclass(frozen=True)
class JevDecision:
    direction: ChoiceAnswer
    intent: ChoiceAnswer
    regime: ChoiceAnswer
    quality: ChoiceAnswer
    latency_ms: int
    input_tokens: int
    model: str


@dataclass(frozen=True)
class StrategyConfig:
    market_limit: int = 5
    trade_usd: float = 25.0
    max_cash_fraction: float = 0.02
    agreement_required: int = 3
    min_direction_p: float = 0.75
    min_intent_p: float = 0.70
    min_quality_p: float = 0.50
    toxic_exit_p: float = 0.75
    interval_s: float = 15.0
    top_depth_levels: int = 5


def _parse_choice(
    answer: object,
    allowed: tuple[str, ...],
    question_name: str,
) -> ChoiceAnswer:
    if not isinstance(answer, dict):
        raise JevError(f"Missing/malformed Jev answer: {question_name}")

    picked = str(answer.get("choice", "")).lower()
    if picked not in allowed:
        raise JevError(
            f"Jev answer {question_name!r} returned invalid choice {picked!r}"
        )

    raw = answer.get("probabilities")
    probabilities: dict[str, float] = {}
    if isinstance(raw, dict):
        for option in allowed:
            try:
                probabilities[option] = max(0.0, float(raw.get(option, 0.0)))
            except (TypeError, ValueError):
                probabilities[option] = 0.0

    total = sum(probabilities.values())
    if total <= 0:
        probabilities = {
            option: 1.0 if option == picked else 0.0 for option in allowed
        }
    else:
        probabilities = {
            option: value / total for option, value in probabilities.items()
        }

    return ChoiceAnswer(picked, probabilities)


class OpenRouterJevClient:
    """Minimal client for OpenRouter's alpha Decisions endpoint."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = OPENROUTER_JEV_MODEL,
        endpoint: str = OPENROUTER_DECISIONS_URL,
        timeout_s: float = 15.0,
        max_attempts: int = 3,
    ) -> None:
        if not api_key.strip():
            raise JevError("OPENROUTER_API_KEY is required")
        self.api_key = api_key.strip()
        self.model = model
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.max_attempts = max(1, max_attempts)
        self.http = httpx.Client(timeout=timeout_s)

    def close(self) -> None:
        self.http.close()

    def evaluate(self, state: dict[str, Any]) -> JevDecision:
        body = {
            "model": self.model,
            "state": state,
            "questions": QUESTIONS,
        }
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.http.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=body,
                )
                if not response.is_success:
                    retryable = (
                        response.status_code in (429, 529)
                        or response.status_code >= 500
                    )
                    error = JevError(
                        f"OpenRouter Decisions HTTP {response.status_code}: "
                        f"{response.text[:300].strip()}"
                    )
                    if not retryable or attempt == self.max_attempts:
                        raise error
                    last_error = error
                    time.sleep(_retry_delay(response, attempt))
                    continue

                payload = response.json()
                answers = payload.get("answers")
                if not isinstance(answers, dict):
                    raise JevError("OpenRouter Jev response has no answers object")

                usage = payload.get("usage")
                input_tokens = 0
                if isinstance(usage, dict):
                    try:
                        input_tokens = int(usage.get("input_tokens", 0) or 0)
                    except (TypeError, ValueError):
                        pass

                return JevDecision(
                    direction=_parse_choice(
                        answers.get("direction"),
                        ("yes", "no", "neutral"),
                        "direction",
                    ),
                    intent=_parse_choice(
                        answers.get("intent"),
                        ("enter", "hold", "exit"),
                        "intent",
                    ),
                    regime=_parse_choice(
                        answers.get("regime"),
                        ("calm", "trending", "toxic"),
                        "regime",
                    ),
                    quality=_parse_choice(
                        answers.get("quality"),
                        ("poor", "weak", "fair", "strong"),
                        "quality",
                    ),
                    latency_ms=round((time.perf_counter() - started) * 1000),
                    input_tokens=input_tokens,
                    model=str(payload.get("model") or self.model),
                )
            except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    break
                time.sleep(min(0.25 * (2 ** (attempt - 1)), 5.0))

        raise JevError(f"OpenRouter Jev request failed: {last_error}")


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    value = response.headers.get("Retry-After")
    if value:
        try:
            return min(max(float(value), 0.0), 5.0)
        except ValueError:
            pass
    return min(0.25 * (2 ** (attempt - 1)), 5.0)


def _best_bid(book: OrderBook) -> float | None:
    return max((level.price for level in book.bids), default=None)


def _best_ask(book: OrderBook) -> float | None:
    return min((level.price for level in book.asks), default=None)


def _midpoint(book: OrderBook) -> float | None:
    bid = _best_bid(book)
    ask = _best_ask(book)
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def _depth(book: OrderBook, levels: int) -> dict[str, float]:
    bids = sorted(book.bids, key=lambda level: level.price, reverse=True)[:levels]
    asks = sorted(book.asks, key=lambda level: level.price)[:levels]
    bid_size = sum(level.size for level in bids)
    ask_size = sum(level.size for level in asks)
    denom = bid_size + ask_size
    imbalance = (bid_size - ask_size) / denom if denom > 0 else 0.0
    return {
        "bid_size": round(bid_size, 6),
        "ask_size": round(ask_size, 6),
        "imbalance": round(imbalance, 6),
    }


def _book_state(book: OrderBook, levels: int) -> dict[str, Any]:
    bid = _best_bid(book)
    ask = _best_ask(book)
    mid = _midpoint(book)
    return {
        "best_bid": bid,
        "best_ask": ask,
        "midpoint": mid,
        "spread": None if bid is None or ask is None else round(ask - bid, 6),
        "top_depth": _depth(book, levels),
    }


def _held_outcome(engine: Engine, market: Market) -> str | None:
    open_positions = [
        position
        for position in engine.db.get_positions_for_market(market.condition_id)
        if position.shares > 1e-9 and not position.is_resolved
    ]
    if len(open_positions) == 1:
        return open_positions[0].outcome.lower()
    if len(open_positions) > 1:
        return "mixed"
    return None


def _held_position(engine: Engine, market: Market, outcome: str):
    return engine.db.get_position(market.condition_id, outcome)


class DecisionGate:
    """Deterministic policy between model judgments and the paper engine."""

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.votes: dict[str, deque[str]] = defaultdict(
            lambda: deque(maxlen=config.agreement_required)
        )

    def reset(self, market_slug: str) -> None:
        self.votes.pop(market_slug, None)

    def choose(
        self,
        market_slug: str,
        decision: JevDecision,
        held: str | None,
    ) -> str:
        direction = decision.direction.choice
        vote = (
            direction
            if direction in ("yes", "no")
            and decision.direction.confidence >= self.config.min_direction_p
            else "neutral"
        )
        votes = self.votes[market_slug]
        votes.append(vote)

        # Fail closed on an unexpected multi-outcome paper position.
        if held == "mixed":
            return "hold"

        # Jev may reduce existing paper risk more easily than it may add risk.
        if held in ("yes", "no"):
            if (
                decision.regime.choice == "toxic"
                and decision.regime.confidence >= self.config.toxic_exit_p
            ):
                return f"exit_{held}"
            if (
                decision.intent.choice == "exit"
                and decision.intent.confidence >= self.config.min_intent_p
            ):
                return f"exit_{held}"
            return "hold"

        agreed = (
            len(votes) == self.config.agreement_required
            and all(item == direction for item in votes)
        )
        if (
            direction in ("yes", "no")
            and agreed
            and decision.direction.confidence >= self.config.min_direction_p
            and decision.intent.choice == "enter"
            and decision.intent.confidence >= self.config.min_intent_p
            and decision.regime.choice != "toxic"
            and decision.quality.choice in ("fair", "strong")
            and decision.quality.confidence >= self.config.min_quality_p
        ):
            return f"enter_{direction}"
        return "hold"


class JevPaperRunner:
    def __init__(
        self,
        engine: Engine,
        jev: OpenRouterJevClient,
        config: StrategyConfig,
        *,
        log_path: Path,
    ) -> None:
        self.engine = engine
        self.jev = jev
        self.config = config
        self.gate = DecisionGate(config)
        self.history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=20))
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def run_cycle(self) -> None:
        markets = self.engine.api.list_markets(
            limit=self.config.market_limit,
            sort_by="liquidity",
        )
        for market in markets:
            self._evaluate_market(market)

    def _evaluate_market(self, market: Market) -> None:
        try:
            yes_token = market.yes_token_id
            no_token = market.no_token_id
        except ValueError:
            return

        try:
            yes_book = self.engine.api.get_order_book(yes_token)
            no_book = self.engine.api.get_order_book(no_token)
            yes_mid = _midpoint(yes_book)
            no_mid = _midpoint(no_book)
            if yes_mid is None or no_mid is None:
                return

            series = self.history[market.slug]
            previous = series[-1] if series else yes_mid
            previous_3 = series[-3] if len(series) >= 3 else previous
            series.append(yes_mid)

            held = _held_outcome(self.engine, market)
            state = {
                "market": {
                    "slug": market.slug,
                    "question": market.question,
                    "liquidity": market.liquidity,
                    "volume": market.volume,
                    "end_date": market.end_date,
                },
                "yes_book": _book_state(
                    yes_book,
                    self.config.top_depth_levels,
                ),
                "no_book": _book_state(
                    no_book,
                    self.config.top_depth_levels,
                ),
                "derived": {
                    "yes_mid_change_1": round(yes_mid - previous, 6),
                    "yes_mid_change_3": round(yes_mid - previous_3, 6),
                    "binary_mid_sum": round(yes_mid + no_mid, 6),
                },
                "paper_position": {
                    "outcome": held or "flat",
                },
            }

            decision = self.jev.evaluate(state)
            action = self.gate.choose(market.slug, decision, held)
            trade = self._execute(market, action)
            self._write_log(
                market=market,
                state=state,
                decision=decision,
                action=action,
                trade=trade,
            )

            print(
                f"{market.slug[:38]:38} "
                f"dir={decision.direction.choice}:"
                f"{decision.direction.confidence:.2f} "
                f"intent={decision.intent.choice}:"
                f"{decision.intent.confidence:.2f} "
                f"regime={decision.regime.choice} "
                f"quality={decision.quality.choice} "
                f"-> {action} ({decision.latency_ms}ms)"
            )
        except (JevError, SimError, httpx.HTTPError) as exc:
            self._write_error(market, exc)
            print(f"{market.slug[:38]:38} error={exc}")

    def _execute(self, market: Market, action: str) -> dict[str, Any] | None:
        if action.startswith("enter_"):
            outcome = action.removeprefix("enter_")
            account = self.engine.get_account()
            amount = min(
                self.config.trade_usd,
                account.cash * self.config.max_cash_fraction,
            )
            if amount < 1.0:
                return None
            result = self.engine.buy(
                market.slug,
                outcome,
                amount,
                order_type="fak",
            )
            self.gate.reset(market.slug)
            return asdict(result.trade)

        if action.startswith("exit_"):
            outcome = action.removeprefix("exit_")
            position = _held_position(self.engine, market, outcome)
            if position is None or position.shares <= 0:
                return None
            result = self.engine.sell(
                market.slug,
                outcome,
                position.shares,
                order_type="fak",
            )
            self.gate.reset(market.slug)
            return asdict(result.trade)

        return None

    def _write_log(
        self,
        *,
        market: Market,
        state: dict[str, Any],
        decision: JevDecision,
        action: str,
        trade: dict[str, Any] | None,
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "market": market.slug,
            "state": state,
            "decision": {
                "direction": asdict(decision.direction),
                "intent": asdict(decision.intent),
                "regime": asdict(decision.regime),
                "quality": asdict(decision.quality),
                "latency_ms": decision.latency_ms,
                "input_tokens": decision.input_tokens,
                "model": decision.model,
            },
            "action": action,
            "trade": trade,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_error(self, market: Market, error: Exception) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "market": market.slug,
            "error": str(error),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Jev against live Polymarket books using paper money only."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".jev-paper"),
        help="Local paper account directory (default: .jev-paper)",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=10_000.0,
        help="Starting paper balance when the account is first created.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=20,
        help="Number of observation cycles; 0 means run until Ctrl+C.",
    )
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--markets", type=int, default=5)
    parser.add_argument("--trade-usd", type=float, default=25.0)
    parser.add_argument("--agreement", type=int, default=3)
    parser.add_argument("--min-direction", type=float, default=0.75)
    parser.add_argument("--min-intent", type=float, default=0.70)
    parser.add_argument(
        "--model",
        default=os.getenv("JEV_MODEL", OPENROUTER_JEV_MODEL),
        help="OpenRouter Jev model (default: ~typesafe/jev-latest).",
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("JEV_OPENROUTER_URL", OPENROUTER_DECISIONS_URL),
        help="OpenRouter Decisions endpoint override.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("jev-runs/decisions.jsonl"),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. "
            "Set an OpenRouter key before starting the Jev experiment."
        )

    config = StrategyConfig(
        market_limit=max(1, args.markets),
        trade_usd=max(1.0, args.trade_usd),
        agreement_required=max(1, args.agreement),
        min_direction_p=min(max(args.min_direction, 0.0), 1.0),
        min_intent_p=min(max(args.min_intent, 0.0), 1.0),
        interval_s=max(1.0, args.interval),
    )
    engine = Engine(args.data_dir)
    jev = OpenRouterJevClient(
        api_key,
        model=args.model,
        endpoint=args.endpoint,
    )

    try:
        try:
            engine.get_account()
        except NotInitializedError:
            engine.init_account(args.balance)

        runner = JevPaperRunner(
            engine,
            jev,
            config,
            log_path=args.log,
        )

        cycle = 0
        while args.cycles == 0 or cycle < args.cycles:
            cycle += 1
            print(f"\n=== cycle {cycle} ===")
            runner.run_cycle()
            if args.cycles == 0 or cycle < args.cycles:
                time.sleep(config.interval_s)

        print("\nPaper balance:")
        print(json.dumps(engine.get_balance(), indent=2))
        print(f"Decision log: {args.log}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(json.dumps(engine.get_balance(), indent=2))
    finally:
        jev.close()
        engine.close()


if __name__ == "__main__":
    main()
