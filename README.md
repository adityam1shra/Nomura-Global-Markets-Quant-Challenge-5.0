# Nomura Global Markets Quant Challenge 5.0

Nomura's Global Markets Quant Challenge is a quantitative finance competition that tests participants on real-world problems spanning pricing, risk, and market microstructure. The fifth edition presented three problems, two of which are included in this repository. Each problem is graded on implementation accuracy, system design, and written documentation.

Participant ID: **240061**

---

## Question 2 — Interest Rate Curve Construction and Pricing

The task involves building a full interest rate curve construction engine from scratch in C++17. Starting from observed market quotes for cash deposits and interest rate swaps, the system bootstraps discount factors across a set of maturity nodes and interpolates between them using two methods: linear interpolation on log-discount factors, and an averaged-quadratic scheme.

The constructed curves are then used to price a new vanilla interest rate swap — computing present value and par swap rate from the perspective of the fixed-rate payer. Beyond pricing, the problem requires an analytical derivation of DV01-style risks: the sensitivity of the swap's present value to each market quote used in calibration. Numerical approximations such as bump-and-revalue are explicitly disallowed.

The final component is a generic, extensible system design that cleanly decouples instrument type, curve calibration logic, and interpolation method. The architecture is expected to support new instruments or interpolation schemes with minimal friction, following object-oriented design principles.

**Files:** `solution.cpp`, `Output.csv`, `documentation.docx`, `Question2.docx`

---

## Question 3 — Market Making and Dynamic Quoting

The problem places the participant in the role of a liquidity provider (LP) operating against six clients over a two-month trade history. The dataset records trade timestamps, sides, volumes, trade prices, mid prices, spreads, and post-trade mid snapshots at intervals of 5 to 30 seconds.

The problem is structured across five tasks of increasing complexity:

- **Task 1** computes adversity profiles for each client — the fraction of trades that turned adverse (i.e., moved against the LP) at each time horizon from 5 to 30 seconds.

- **Task 2** extends this to expected PnL per trade and recommends a minimum half-spread per client such that the LP's aggregate expected PnL becomes non-negative.

- **Task 3** trains a probabilistic model to predict, at trade time, the likelihood that a given trade will be adverse at a specified horizon. Model performance is evaluated on accuracy, precision, recall, and log-loss across train, validation, and test splits.

- **Task 4** uses the adversity model to optimise an externalization strategy. A trade is externalized (i.e., risk is offloaded to a third party at zero cost) when the predicted adversity probability exceeds a threshold. The task identifies the threshold that maximises validation-set PnL and reports final test-set results.

- **Task 5** is the core challenge: designing a dynamic quoting function that adjusts bid and ask half-spreads in real time based on current inventory, realized volatility, the adversity score of the incoming client, and a momentum signal. The strategy is evaluated on a Sharpe-like score against a hidden test stream that includes undisclosed regime shifts.

**Files per task:** Python solution (`taskN.py`), results (`taskN_results.csv`), write-up (`taskNdoc.pdf`)

---

## Repository Structure

```
Question 2/
    solution.cpp
    Output.csv
    documentation.docx
    Question2.docx

Question3/
    Question3.pdf
    task1/
    task2/
    task3/
    task4/
    task5/
```
