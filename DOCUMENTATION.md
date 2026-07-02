# Project Documentation - Multi-LLM Collaborative Debate System

This document explains how the system is built, every design decision and why it was made,
what every function does, and which course topics the project covers.

---

## 1. What the system does (short version)

Four LLM agents (four different models from two providers) get a problem. First, each agent self-assesses
whether it would rather be a Solver or the Judge for this specific problem (Stage 0), and a
deterministic algorithm assigns the roles (Stage 0.5). The three Solvers solve independently
(Stage 1), review each other's solutions with structured critique (Stage 2), and refine their
solutions in response to the critiques - accepting valid ones and defending against wrong ones
(Stage 3). Finally the Judge reads everything and picks the winner (Stage 4). The winner's
refined answer is the system's answer.

The whole point: a single LLM hallucinates and makes reasoning slips. Different models have
DIFFERENT failure modes, so adversarial cross-review plus an independent verification step
catches errors that any single model would miss.

---

## 2. Architecture and file layout

```
problems.json            dataset: 25 problems, 5 categories, verifiable answers
verify_ground_truths.py  independent Python computation of every ground truth
01_debate_system.ipynb   all pipeline code + full run + baselines (produces results/)
02_evaluation.ipynb      metrics + plots (consumes results/, produces plots/)
results/                 debate_results.json, baseline_results.json, metrics_summary.csv
plots/                   4 generated plots
.env                     2 API keys, OpenAI + Anthropic (gitignored, never committed)
requirements.txt         openai, pydantic, python-dotenv, matplotlib, pandas, jupyter
```

Two notebooks instead of one: the expensive part (API calls) and the cheap part (analysis) are
separated, so you can re-run the whole evaluation and re-style plots without paying for a single
API call. They communicate through plain JSON files on disk.

---

## 3. Design decisions and the reasoning behind them

### 3.1 Four different models through one SDK - deliberately strong + weak

The agents are gpt-4.1-mini and gpt-3.5-turbo (OpenAI), claude-sonnet-4-5 and claude-haiku-4-5
(Anthropic): from each provider, one capable modern model and one weaker/older or smaller model.

Why include weak models on purpose? Because of a real finding from an earlier iteration: with
four frontier-tier agents, every metric saturated - all solvers were right, everyone agreed,
accuracy was ~100% across the board, and the debate machinery had literally nothing to do. That
result technically "works" but demonstrates nothing. Adding gpt-3.5-turbo (a 2023-era model) and
keeping haiku (the small Claude tier) reintroduces realistic errors, which is exactly what makes
the system's metrics meaningful: reviews catch actual mistakes, refinement fixes (or fails to
fix) actual answers, and the judge faces actual disagreements. This mirrors the real production
motivation for debate systems - you use them precisely because your models are fallible.

Diversity still holds: the 4 models are 4 separately trained model families across 2 training
lineages, with very different capability levels - which produces genuinely uncorrelated errors.

History of this decision: the project originally ran with 4 separate providers - OpenAI,
Anthropic, Google Gemini and Meta's Llama 3.3 70B served by Groq. Both free-tier keys proved
unreliable in practice: Gemini kept exhausting its daily quota mid-run (see 3.6 for the fallout),
and Groq hit similar rate/quota problems. Both slots were replaced with models from the two
paid, reliable providers. This is itself a realistic production decision: an unreliable
dependency is worse than a slightly less diverse ensemble, and the assignment explicitly allows
even a single provider with different models.

Decision: call all agents through the OpenAI Python SDK. Anthropic exposes an OpenAI-compatible
REST endpoint, so the only per-agent configuration is `base_url` + API key + model name.
One `ask()` function serves everything; there is no provider-specific code anywhere else in the
pipeline. Swapping an agent (as happened several times) is a one-dict-entry change.

### 3.2 Structured outputs everywhere, with a fallback path

Every stage returns JSON validated against a Pydantic schema (like in the course lectures on
structured outputs). The OpenAI models and Anthropic support `response_format` with a JSON schema
natively through `client.chat.completions.parse()` - the API guarantees schema-valid JSON.

Three of the four models support this natively. gpt-3.5-turbo predates structured outputs
(it only has plain JSON mode), and the original Llama agent had the same limitation - so
`ask()` has a fallback path that is actively used in every run:

1. the Pydantic schema is serialized and injected into the system prompt
2. the model is forced into JSON mode with `response_format={"type": "json_object"}`
3. the output is parsed with `json.loads` and validated with `schema.model_validate(...)`
4. if validation fails, the retry includes the exact validation error in the prompt so the model
   can fix it

Two real bugs found during development, both good examples of why native structured outputs exist:

- **schema echo**: when you just paste a JSON schema into the prompt, weaker models sometimes
  return the SCHEMA itself instead of an instance of it. Fix: explicitly demand "actual values,
  NOT the schema itself" and list the expected top-level keys.
- **type drift**: gpt-3.5-turbo likes to put a nested object or a bare number where the schema
  wants a string (e.g. `final_answer: {"payment": 843.86, "interest": 51894}`). Instead of
  failing validation 6 times in a row, the fallback now coerces such values (dict/list ->
  JSON string, number -> string) before Pydantic validation; genuinely missing/null fields
  still fail and trigger the error-feedback retry.

### 3.3 Deterministic role assignment (Stage 0.5)

The assignment requires a deterministic algorithm on top of the self-assessments. The rule:

1. Judge = the agent with the highest self-reported `judge_confidence`; ties are broken by a
   fixed agent order (openai, openai-35, claude, claude-haiku), so the result is always reproducible
2. the remaining three agents become Solvers, ordered by `solver_confidence` descending
   (solver_1 = most confident), same tie-breaking

No randomness anywhere: same self-assessments in, same role distribution out.

### 3.4 Prompt design against known failure modes

Three specific behaviors had to be engineered away in the prompts:

- **Reviewers inventing errors.** A reviewer asked to "find errors" will find errors, even in a
  perfect solution, just to be useful. The review prompt explicitly says "do NOT invent errors
  that are not there - if the solution is fully correct, say so".
- **Solvers folding under criticism (sycophancy).** By default a model that receives criticism
  agrees with it, even when the criticism is wrong, and "fixes" a correct answer into a wrong
  one. The refinement prompt explicitly requires accepting OR rejecting each critique, and says
  "if your original reasoning was correct, defend it".
- **The judge counting votes instead of thinking.** The judge prompt says "do not just count
  votes or trust confidence scores - verify the reasoning of each refined solution yourself".

### 3.5 Grading free-text answers with a separate LLM grader

Answers come back as free text: "1/3", "one third", "the probability is 33.3%". Exact string
matching is useless. A separate grader model (gpt-4.1-mini at temperature 0, NOT part of the
debate) compares each candidate answer with the ground truth and returns a structured
correct/incorrect verdict. The same technique (LLM-as-judge for evaluation) also powers answer
clustering: the majority-voting baseline and the consensus metric need to know whether "7.5"
and "7.5 degrees" are the same answer.

Why this is acceptable: equivalence checking against a KNOWN ground truth is a far easier task
than solving the problem, so a small model at temperature 0 is reliable at it - IF the ground
truths and the grading instruction are written carefully. That "if" was learned the hard way:

**The grading bug.** In the first version of the hard dataset, ground truths contained the
derivation ("By Fermat, 7^10 = 1 (mod 11), 222 = 22*10 + 2, so ... Answer: 5") and the grader
was told that with multiple required values, ALL must match. The grader then treated the
derivation fragments as required values - so a solver that answered a bare, perfectly correct
"5" was graded WRONG for not reproducing the Fermat steps. This silently deflated several
scores (two problems showed 0/3 solvers correct when all three were actually right). The fix
had two halves: (1) ground truths were rewritten to contain ONLY the required final values, and
(2) the grading prompt now says explicitly to judge final answer VALUES only - derivations are
never required and a bare correct value counts as fully correct. The fixed grader was then
regression-tested against 13 hand-picked cases from the buggy run (previously mis-graded correct
answers, plus genuinely wrong answers that must stay wrong) - all 13 grade correctly now.

Lesson worth stating out loud: when your evaluator is an LLM, your evaluation is code too, and
it needs testing exactly like code. Grader mistakes do not crash anything - they silently
corrupt your metrics.

### 3.6 Checkpointing and fault tolerance

A full run is 600+ API calls to external services. Something WILL fail (rate limit, network
blip, an overloaded provider). Decisions:

- results are saved to `results/debate_results.json` after EVERY problem; re-running the loop
  skips finished problems, so a crash costs one problem, not the whole run
- every API call retries up to 6 times with exponential backoff capped at 60s - the cap matters
  because rate-limited providers typically throttle per minute, so short retries would all burn
  inside the same rate window (this was very visible with the free-tier providers the project
  started with)
- validation failures (fallback path) retry after only 2s, because they are not rate problems
- a failed problem is caught, logged and skipped instead of killing the loop - the console
  says "FAILED ... re-run this cell to retry", and because of the checkpoint, re-running the
  loop cell retries ONLY the missing problems. A run that ends with "22/25 done" is therefore
  not broken; it is one cell re-run away from 25/25.
- the grader calls have their own retry loop too (a transient network timeout in a grading call
  once killed an otherwise perfect debate - graders must be as fault-tolerant as agents)
- every client is created with `timeout=120, max_retries=0`. This was learned the hard way:
  the OpenAI SDK's DEFAULT is a 10-minute timeout with internal retries, so when the original
  Gemini agent's free-tier daily quota died mid-run, the pipeline did not fail - it silently
  stalled for half an hour. An explicit short timeout plus our own retry loop makes failures
  visible and fast. Gemini was subsequently dropped entirely (see the decision log in section 7),
  because a dependency that dies mid-run is worse than a slightly less diverse ensemble.

### 3.7 Parallelism inside stages

Calls inside one stage are independent (3 solvers don't need to wait for each other), so they run
concurrently with `ThreadPoolExecutor` (max 6 workers). Stages themselves stay sequential because
each stage consumes the previous stage's output. This cut the per-problem wall time roughly 3x.

### 3.8 Dataset design

25 problems, 5 categories, exam-style. The first version of the dataset used short puzzle-type
questions (clock angles, snail in a well, Monty Hall variants) with one-word/one-number answers.
It turned out to be too easy for modern models - even the initial solutions were nearly always
right and every metric saturated. The redesigned dataset follows three rules:

1. **multi-step derivations, not lookups** - every problem hands the model concrete numbers and
   demands a computation carried through several steps (annuity formula, energy balance,
   characteristic equation, CRT reconstruction). Memorized trivia does not produce the answer;
   only correct step-by-step arithmetic does.
2. **multi-part answers** - most problems require 2-3 values (radius AND height AND area;
   probability AND expected duration; payment AND total interest). This makes
   partially-correct answers common, which is exactly what peer review exists to catch, and it
   makes the strict grader meaningful (ALL required values must be right).
3. **machine-verified ground truth** - every answer in problems.json was computed independently
   with Python before being added; the script ships with the repo as `verify_ground_truths.py`
   (analytic formulas plus brute-force/simulation cross-checks where applicable).

| category | what it tests | example |
|---|---|---|
| optimization | calculus optimization, related rates | closed can of 500 cm^3 with minimal surface |
| probability_expectation | state equations, expectation decompositions | E[flips] to see HH vs HT and why they differ |
| applied_physics | multi-step numeric derivations | spring launch -> energy conservation -> height |
| finance | annuity/discounting formulas with exact numbers | 15-year mortgage payment + total interest |
| discrete_math | modular arithmetic, recurrences | CRT system, closed form of a(n)=3a(n-1)-2a(n-2) |

### 3.9 Baselines

Two baselines, as required:

- **single-LLM**: each of the 4 models is asked each problem once, with the same solver prompt
  and temperature as in the debate - this isolates the effect of the debate machinery itself
- **majority voting**: take the 3 independent Stage-1 solutions and pick the majority answer
  (clustered semantically by the grader model, tie -> first). This baseline is computed from the
  debate's own Stage-1 outputs, which is both cheaper and a fairer comparison: voting and debate
  start from the exact same three initial solutions, so any difference is caused purely by the
  review/refine/judge stages.

### 3.10 Metrics (Phase 3)

- **overall accuracy** - is the system's final answer correct
- **improvement rate** - problems where refined answers are strictly better than initial ones
  (count of correct refined > count of correct initial)
- **consensus rate** - all 3 refined answers are semantically equivalent
- **judge accuracy on disagreements** - among problems where solvers still disagree after
  refinement, how often the judge picks a correct answer; also reported conditional on "a correct
  option existed", because the judge cannot win if all three refined answers are wrong
- small robustness details: `clamp01()` normalizes confidence values because some models
  occasionally return 95 instead of 0.95

---

## 4. Function reference (01_debate_system.ipynb)

| function | what it does |
|---|---|
| `ask(agent, system_prompt, user_prompt, schema, temperature, retries)` | one structured API call to any provider; native `parse()` path or JSON-mode fallback with Pydantic validation; retries with backoff |
| `ask_grader(system_prompt, user_prompt, schema, retries)` | structured call to the grader model, always temperature 0, with its own retry loop |
| `run_parallel(tasks)` | runs a list of zero-argument functions concurrently, returns results in order |
| `clamp01(x)` | normalizes confidence values to [0, 1] (fixes 95 -> 0.95 style outputs) |
| `assess_roles(problem)` | Stage 0: asks all 4 agents for role preference + confidences (parallel) |
| `assign_roles(assessments)` | Stage 0.5: deterministic judge/solver assignment with fixed tie-breaking |
| `solve_independently(problem, solvers)` | Stage 1: 3 independent solutions (parallel) |
| `peer_review(problem, solutions, solver_agents)` | Stage 2: each solver reviews the other two -> 6 structured reviews (parallel) |
| `format_reviews(review_list)` | renders structured reviews into readable text for the next prompts |
| `refine_solutions(problem, solutions, reviews, solver_agents)` | Stage 3: each solver responds to every critique and refines (parallel) |
| `judge_solutions(problem, judge_agent, solutions, reviews, refinements)` | Stage 4: judge sees everything, returns winner + confidence + reasoning |
| `grade(problem, candidate)` | LLM grader: is the candidate answer equivalent to ground truth |
| `majority_vote(problem, answers)` | clusters answers semantically, returns majority + whether all agree |
| `run_debate(problem)` | full pipeline for one problem, returns one JSON-serializable record with everything graded |
| `load_checkpoint(path)` / `save_checkpoint(path, data)` | JSON checkpoint helpers |

Schemas (Pydantic): `RoleAssessment`, `Solution`, `ReviewError`, `PeerReview`,
`CritiqueResponse`, `Refinement`, `JudgeVerdict`, `GradeResult`, `MajorityResult`.

The evaluation notebook (`02_evaluation.ipynb`) flattens the records into a pandas DataFrame,
computes all metrics, and generates 4 plots (accuracy vs baselines, accuracy by category,
effect of refinement, role/win distribution). Plots use the Okabe-Ito colorblind-safe palette.

---

## 5. Course topics covered by this project

- **LLM APIs** - chat completions, system vs user prompts, temperature, multiple providers,
  OpenAI-compatible endpoints
- **Structured outputs** - Pydantic schemas as `response_format`, `.parse()`, and the manual
  parse-and-validate fallback (what we did before structured outputs existed)
- **Prompt engineering** - role prompts per stage, fighting sycophancy and invented critiques,
  forcing step-by-step reasoning
- **Multi-step LLM pipelines / orchestration** - output of one call becomes input of the next,
  5 stages deep, with state passed between stages
- **LLM-as-judge / LLM-based evaluation** - both inside the system (Stage 4) and outside it
  (the grader)
- **Ensembling and self-consistency** - majority voting baseline vs debate
- **API key management** - .env + python-dotenv, keys never in code or git
- **Production concerns** - retries, exponential backoff, per-minute rate limits, checkpointing,
  parallel calls, cost/latency awareness

---

## 6. Known limitations and possible extensions

- **Single debate round.** Review -> refine happens once. A natural extension is looping
  Stage 2-3 until consensus or a round limit.
- **No tools.** Solvers do arithmetic in their heads. Giving them a code interpreter would
  probably fix most calculation slips (but then the project would be about tool use, not debate).
- **Judge is a single point of failure.** When solvers disagree, one model decides. An extension:
  a judge panel with its own vote, or weighting the judge's verdict by solver confidences.
- **Possible training-data contamination.** The problem TYPES are standard textbook material
  (they have to be, to have verifiable answers). The defense is that the specific numbers force
  a fresh computation - a model that has seen "minimize the can surface" a thousand times still
  has to compute (250/pi)^(1/3) for THIS volume - and the observed failures are arithmetic
  slips, not retrieval gaps. A fully contamination-free dataset would need procedurally
  generated novel problems.
- **The grader is an LLM.** Its verdicts are regression-tested (13 hand-picked cases) and were
  spot-checked, but a fully rigorous evaluation would force exact answer formats (e.g.
  numeric-only fields) to allow programmatic grading. The grading bug described in 3.5 shows
  how real this risk is.

---

## 7. Decision log - how the project actually evolved

Kept honest on purpose: several of these decisions were forced by things breaking, and each
breakage taught something about running LLM systems in practice.

1. **v1 - four providers.** Agents: gpt-4.1-mini (OpenAI), claude-sonnet-4-5 (Anthropic),
   gemini-2.5-flash (Google), llama-3.3-70b on Groq (Meta weights). Maximum lineage diversity,
   all through one SDK via OpenAI-compatible endpoints. Dataset: 25 puzzle-style questions
   (clock angles, counting, trap word problems, probability paradoxes).
2. **Gemini dropped.** Its free-tier daily quota exhausted itself mid-run, and thanks to the
   SDK's default 10-minute timeout the pipeline silently STALLED instead of failing. Two fixes
   came out of it: explicit `timeout=120, max_retries=0` on every client (fail fast, retry
   deliberately), and eventually removing Gemini - first downgrading to flash-lite (separate
   quota bucket), then replacing the slot with a second OpenAI model when quota problems
   continued. Lesson: failures that raise are easy; failures that hang are dangerous.
3. **Groq/Llama dropped.** Same class of problem (free-tier rate and quota walls mid-run), plus
   it was the model that needed the JSON fallback path. Replaced with a second Anthropic model
   (claude-haiku-4-5). At this point the lineup was 2 OpenAI + 2 Anthropic, all reliable paid
   endpoints. Lesson: an unreliable dependency costs more than the diversity it adds - and the
   assignment explicitly allows one provider with multiple models.
4. **Everything saturated at 100%.** A full run with the strong 2x2 lineup on the puzzle dataset
   produced ~100% accuracy for every single model, voting AND the debate. Technically working,
   scientifically empty - no errors means nothing for the debate to demonstrate. Two changes:
   the dataset was redesigned into multi-step exam problems with multi-part answers (see 3.8),
   and one slot per provider was deliberately downgraded (gpt-3.5-turbo in; it also revived the
   fallback path with real traffic). Lesson: evaluation design is as important as system design;
   a benchmark everyone aces measures nothing.
5. **The grading bug.** Spot-checking the first hard-dataset run revealed the grader was
   requiring derivation fragments from the ground truth as if they were answer values, marking
   objectively correct answers wrong (see 3.5). Ground truths were rewritten to values-only, the
   grading prompt was rewritten to judge values only, and the fix was regression-tested against
   13 cases from the buggy run. Lesson: an LLM evaluator is code - test it like code, because
   its failures are silent.

The final configuration: gpt-4.1-mini + gpt-3.5-turbo + claude-sonnet-4-5 + claude-haiku-4-5,
exam-style dataset, values-only ground truths, tested grader, fault-tolerant checkpointed
pipeline.
