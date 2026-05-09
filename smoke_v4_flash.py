#!/usr/bin/env python3
"""Smoke-test SGLang serving DeepSeek-V4-Flash at http://localhost:30000.

Verifies in order:
  1. GET  /v1/models           returns 200 and lists the served model id.
  2. POST /v1/completions      with prompt='hi', max_tokens=1 returns text.
  3. POST /v1/score            accepts label_token_ids and returns
                               a list-of-one row of length len(label_token_ids),
                               with non-negative entries summing to <= 1.0,
                               and at least one strictly positive.

Prints "PASS" if all checks pass; "FAIL" with details otherwise.
Exit 0 on PASS, 1 on FAIL.
"""

import json
import sys
import urllib.error
import urllib.request

URL = "http://localhost:30000"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
# DeepSeek-V4 special tokens (full-width pipe U+FF5C, lower-one-eighth-block U+2581).
QUERY = (
    "<｜begin▁of▁sentence｜>"
    "<｜User｜>Pick a digit."
    "<｜Assistant｜></think>The digit is 0."
)
LABEL_TOKEN_IDS = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
TIMEOUT_S = 120


def _http(method: str, path: str, payload: dict | None = None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        f"{URL}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return r.status, json.loads(r.read())


def main() -> int:
    failures: list[str] = []

    # ---- 1. /v1/models ----------------------------------------------------
    print("[smoke] GET /v1/models")
    try:
        status, body = _http("GET", "/v1/models")
    except urllib.error.URLError as e:
        print(f"[smoke] FAIL: cannot reach {URL}: {e}")
        return 1
    if status != 200:
        failures.append(f"models: HTTP {status}")
    else:
        ids = [m.get("id") for m in body.get("data", [])]
        print(f"[smoke]   served model ids: {ids}")
        if not ids:
            failures.append("models: empty data")

    # ---- 2. /v1/completions ----------------------------------------------
    print("[smoke] POST /v1/completions  (prompt='hi', max_tokens=1)")
    try:
        status, body = _http(
            "POST",
            "/v1/completions",
            {"model": MODEL, "prompt": "hi", "max_tokens": 1, "temperature": 0.0},
        )
    except urllib.error.HTTPError as e:
        body = json.loads(e.read() or b"{}") if e.fp else {}
        status = e.code
    if status != 200:
        failures.append(f"completions: HTTP {status}: {json.dumps(body)[:300]}")
    else:
        text = body.get("choices", [{}])[0].get("text", "")
        print(f"[smoke]   completion text: {text!r}")

    # ---- 3. /v1/score with label_token_ids -------------------------------
    print("[smoke] POST /v1/score  (with label_token_ids)")
    try:
        status, body = _http(
            "POST",
            "/v1/score",
            {
                "model": MODEL,
                "query": QUERY,
                "items": [""],
                "label_token_ids": LABEL_TOKEN_IDS,
                "apply_softmax": False,
            },
        )
    except urllib.error.HTTPError as e:
        body = json.loads(e.read() or b"{}") if e.fp else {}
        status = e.code
    print(f"[smoke]   /v1/score response keys: {list(body.keys())}")
    if status != 200:
        failures.append(f"score: HTTP {status}: {json.dumps(body)[:500]}")
    else:
        scores = body.get("scores")
        if not isinstance(scores, list) or len(scores) != 1:
            failures.append(f"score: expected list-of-one, got {type(scores).__name__} len={len(scores) if isinstance(scores, list) else 'N/A'}")
        else:
            row = scores[0]
            if not isinstance(row, list):
                failures.append(f"score: scores[0] is not a list: {type(row).__name__}")
            elif len(row) != len(LABEL_TOKEN_IDS):
                failures.append(f"score: row len {len(row)} != len(label_token_ids) {len(LABEL_TOKEN_IDS)}")
            elif any((not isinstance(x, (int, float))) or x < 0 for x in row):
                failures.append(f"score: non-numeric or negative entry in row: {row}")
            elif sum(row) > 1.0 + 1e-3:
                failures.append(f"score: probabilities sum {sum(row):.6f} > 1.0")
            elif all(x == 0 for x in row):
                failures.append(f"score: all probabilities are zero: {row}")
            else:
                print(f"[smoke]   scores[0] = {row}")
                print(f"[smoke]   sum(scores[0]) = {sum(row):.6f}")
                print(f"[smoke]   max  = {max(row):.6f}  argmax token = {LABEL_TOKEN_IDS[row.index(max(row))]}")

    if failures:
        print("[smoke] FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
