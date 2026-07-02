# Multi-LLM Collaborative Debate System

Final project for the Applied LLM Systems course.

Three LLM agents solve a problem independently, review each other's solutions with structured
critique, refine their answers based on the peer feedback, and a fourth LLM (the Judge) verifies
everything and picks the best final answer. The idea is to fight hallucinations and reasoning
errors with diverse perspectives and adversarial review instead of trusting one model.

## Pipeline

```
                 problem
                    |
   Stage 0   role self-assessment (all 4 agents rate themselves as Solver / Judge)
                    |
   Stage 0.5 deterministic role assignment -> 3 Solvers + 1 Judge
                    |
   Stage 1   3 independent solutions (no communication between solvers)
                    |
   Stage 2   peer review: each solver reviews the other two (6 structured reviews)
                    |
   Stage 3   refinement: every critique is accepted+fixed or rejected+defended
                    |
   Stage 4   the Judge verifies originals + reviews + refined solutions, picks a winner
                    |
              final answer = winner's refined answer
```

## Agents - 2 providers, strong + weak model from each

The 4 agents are 4 different models - from each provider one strong modern model and one
deliberately weaker/older model - called through one and the same OpenAI Python SDK (Anthropic
exposes an OpenAI-compatible endpoint, so the only per-agent difference is `base_url`, key and
model name):

| agent | provider | model | tier | structured outputs |
|---|---|---|---|---|
| openai | OpenAI | gpt-4.1-mini | mid (2025) | native json_schema |
| openai-35 | OpenAI | gpt-3.5-turbo | weak (2023) | fallback: JSON mode + Pydantic validation |
| claude | Anthropic | claude-sonnet-4-5 | strong | native json_schema |
| claude-haiku | Anthropic | claude-haiku-4-5 | small/fast | native json_schema |

The weak models are a deliberate design choice: in an early version all four agents were
frontier-tier models and every metric saturated at ~100% accuracy - the debate had nothing to do.
With a strong/weak mix the mechanism becomes measurable: weak solvers make real errors, reviews
catch real things, refinement fixes real things, and the judge resolves real disagreements.

Historical note: the original lineup was 4 separate providers (OpenAI, Anthropic, Google Gemini,
Llama on Groq), but the Gemini and Groq free-tier keys kept dying mid-run (daily quotas, rate
limits), so both slots were replaced with models from the two reliable providers. The assignment
explicitly allows using a single provider with different models.

Which agent becomes the Judge is decided per problem by the Stage 0.5 algorithm (highest
self-reported judge confidence wins, fixed tie-breaking, fully deterministic).

A separate grader model (gpt-4.1-mini, not part of the debate) compares free-text answers with
the ground truth so that "1/3", "0.333" and "one third" all count as the same answer.

## Dataset

`problems.json` - 25 exam-style problems with verifiable answers. Design goals: every problem is
a multi-step derivation with concrete numbers (cannot be answered from memorized trivia), most
answers have 2-3 required parts (so partially-right answers are common and reviewers have
something to catch), and every ground truth was computed independently with Python before being
added (`verify_ground_truths.py`).

- **optimization** (5): minimal-surface cylinder, fence along a river, open-top box,
  constrained product maximization, related-rates ladder
- **probability_expectation** (5): HH vs HT expected waiting times, gambler's ruin (probability
  AND duration), coupon collector, expected maximum of two dice, expected position of first ace
- **applied_physics** (5): incline with friction, projectile from a cliff, spring launch with
  energy conservation, max speed on an unbanked curve, RC circuit charging
- **finance** (5): mortgage payment and total interest, NPV with annuity factor, exact doubling
  time vs Rule of 72, future value of monthly savings, effective annual rates
- **discrete_math** (5): CRT system of 3 congruences, Fermat's little theorem, Euler's totient,
  linear recurrence via characteristic equation, base-7 addition with carries

## Files

```
problems.json            the 25-problem dataset (question + ground truth answer)
verify_ground_truths.py  computes every ground truth independently (dataset sanity check)
01_debate_system.ipynb   the whole system + runs it on the dataset + baselines
02_evaluation.ipynb      metrics and plots (Phase 3)
DOCUMENTATION.md         full write-up: design decisions, function reference, limitations
results/                 raw results (debate_results.json, baseline_results.json, metrics_summary.csv)
plots/                   generated plots
requirements.txt         dependencies
```

## How to run

1. `pip install -r requirements.txt`
2. create a `.env` file in the project root with the four keys:
   ```
   OPENAI_API_KEY=...
   ANTHROPIC_API_KEY=...
   ```
3. run `01_debate_system.ipynb` top to bottom. It makes ~30 API calls per problem, so the full
   run takes a while. Results are checkpointed to `results/debate_results.json` after every
   problem - if something crashes (rate limits happen on free tiers) just re-run the cell and it
   continues where it stopped.
4. run `02_evaluation.ipynb` to get all metrics and the plots.

## Results

(to be filled from results/metrics_summary.csv after the full run)

See `02_evaluation.ipynb` and `plots/` for the full analysis, and `DOCUMENTATION.md` for the
design decisions.
