"""Offline tests for the OpenRouter Jev paper experiment."""

from __future__ import annotations

import httpx
import pytest

from examples.jev_paper import (
    OPENROUTER_DECISIONS_URL,
    OPENROUTER_JEV_MODEL,
    ChoiceAnswer,
    DecisionGate,
    JevDecision,
    JevError,
    OpenRouterJevClient,
    StrategyConfig,
    _parse_choice,
)


def _decision(
    *,
    direction: str = "yes",
    direction_p: float = 0.9,
    intent: str = "enter",
    intent_p: float = 0.9,
    regime: str = "calm",
    regime_p: float = 0.9,
    quality: str = "strong",
    quality_p: float = 0.9,
) -> JevDecision:
    def answer(choice: str, probability: float, options: tuple[str, ...]) -> ChoiceAnswer:
        remainder = (1.0 - probability) / max(len(options) - 1, 1)
        probs = {item: remainder for item in options}
        probs[choice] = probability
        return ChoiceAnswer(choice, probs)

    return JevDecision(
        direction=answer(direction, direction_p, ("yes", "no", "neutral")),
        intent=answer(intent, intent_p, ("enter", "hold", "exit")),
        regime=answer(regime, regime_p, ("calm", "trending", "toxic")),
        quality=answer(quality, quality_p, ("poor", "weak", "fair", "strong")),
        latency_ms=12,
        input_tokens=123,
        model="typesafe/jev-1.13",
    )


def test_openrouter_defaults():
    assert OPENROUTER_DECISIONS_URL == "https://openrouter.ai/api/alpha/decisions"
    assert OPENROUTER_JEV_MODEL == "~typesafe/jev-latest"


def test_client_requires_openrouter_key():
    with pytest.raises(JevError, match="OPENROUTER_API_KEY"):
        OpenRouterJevClient("")


def test_choice_parser_normalizes_probabilities():
    parsed = _parse_choice(
        {
            "choice": "yes",
            "probabilities": {"yes": 7, "no": 2, "neutral": 1},
        },
        ("yes", "no", "neutral"),
        "direction",
    )
    assert parsed.choice == "yes"
    assert parsed.confidence == pytest.approx(0.7)
    assert sum(parsed.probabilities.values()) == pytest.approx(1.0)


def test_client_posts_decisions_schema(monkeypatch):
    client = OpenRouterJevClient("sk-or-test", max_attempts=1)
    captured = {}
    response = httpx.Response(
        200,
        request=httpx.Request("POST", OPENROUTER_DECISIONS_URL),
        json={
            "model": "typesafe/jev-1.13",
            "answers": {
                "direction": {
                    "type": "choice",
                    "choice": "yes",
                    "probabilities": {"yes": 0.8, "no": 0.1, "neutral": 0.1},
                },
                "intent": {
                    "type": "choice",
                    "choice": "enter",
                    "probabilities": {"enter": 0.8, "hold": 0.1, "exit": 0.1},
                },
                "regime": {
                    "type": "choice",
                    "choice": "calm",
                    "probabilities": {"calm": 0.8, "trending": 0.1, "toxic": 0.1},
                },
                "quality": {
                    "type": "choice",
                    "choice": "strong",
                    "probabilities": {
                        "poor": 0.05,
                        "weak": 0.05,
                        "fair": 0.1,
                        "strong": 0.8,
                    },
                },
            },
            "usage": {"input_tokens": 456, "output_tokens": 0},
        },
    )

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return response

    monkeypatch.setattr(client.http, "post", fake_post)
    try:
        result = client.evaluate({"market": {"question": "test"}})
    finally:
        client.close()

    assert captured["url"] == OPENROUTER_DECISIONS_URL
    assert captured["headers"]["Authorization"] == "Bearer sk-or-test"
    assert captured["json"]["model"] == OPENROUTER_JEV_MODEL
    assert set(captured["json"]["questions"]) == {
        "direction",
        "intent",
        "regime",
        "quality",
    }
    assert result.direction.choice == "yes"
    assert result.direction.confidence == pytest.approx(0.8)
    assert result.input_tokens == 456


def test_gate_requires_three_high_confidence_agreements():
    gate = DecisionGate(StrategyConfig(agreement_required=3))
    decision = _decision()

    assert gate.choose("market-a", decision, None) == "hold"
    assert gate.choose("market-a", decision, None) == "hold"
    assert gate.choose("market-a", decision, None) == "enter_yes"


def test_gate_blocks_entry_in_toxic_regime():
    gate = DecisionGate(StrategyConfig(agreement_required=1))
    decision = _decision(regime="toxic", regime_p=0.9)
    assert gate.choose("market-a", decision, None) == "hold"


def test_gate_allows_risk_reducing_toxic_exit():
    gate = DecisionGate(StrategyConfig())
    decision = _decision(
        intent="hold",
        regime="toxic",
        regime_p=0.9,
    )
    assert gate.choose("market-a", decision, "yes") == "exit_yes"
