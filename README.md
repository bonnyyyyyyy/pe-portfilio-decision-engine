# Private Equity Portfolio Decision Engine

**SKELAR × Horizon Capital × Mastercard Case Challenge 3.0**

Interactive, reproducible Streamlit application for 24-month portfolio exit and incremental fund cash analysis. The three companies are illustrative case businesses, **not actual investments or verified fund forecasts**.

> **Illustrative investment model — results are not verified fund forecasts.** Historical case facts and forward analyst assumptions are explicitly separate. Transaction terms and the source probabilities have not been independently verified.

## Quick start

Requirements: **Python 3.11+** and internet access for the initial dependency installation.

```bash
git clone <your-github-repository-url>
cd pe-portfolio-decision-engine
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

Open the URL printed by Streamlit (normally `http://localhost:8501`). The application has **Portfolio Overview & Sensitivity**, **Monte Carlo Simulation** and **Capital Allocation & Strategy Comparison** tabs, plus an expandable **Sources & Model Audit** panel.

## Data and source hierarchy

| Input | Location in the PDF | Status |
| --- | --- | --- |
| Historical company financials, ownership and mandate | Team PDF pp. 7–12, summarizing official case | Official case facts / derived |
| Stage 11 | pp. 66–68 | Original analysis; error-review only |
| C8 No-Action market states | p. 346 | Hypothetical analyst scenarios |
| C8 fully successful funded interventions | p. 347 | Hypothetical analyst scenarios |
| C8 execution fractions and common execution state | pp. 347–348 | Hypothetical execution outcomes |
| C8 market-conditional execution probabilities | pp. 351–353 | **Uncalibrated** probabilities |
| C9 $24m IT pricing-date EBITDA and cash waterfall | pp. 371–376 | Corrected but **unverified** Base assumptions |

PDF source is included at `data/Case Challenge (10).pdf`. Input JSON files are:

- `data/companies.json`: case historical facts, separate from M24 forecasts.
- `data/c9_base.json`: transaction-pricing-date C9 valuation basis.
- `data/c8_scenarios.json`: **different** C8 paired market assumptions, funded success assumptions, and conditionally weighted execution factors.

The initial C9 **No-Action** gross fund proceeds are $119.800m, plus $15.000m unspent cash = **$134.800m**. C9 **IT Full + Security Pilot** gross proceeds are $149.400m, plus $3.000m retained cash = **$152.400m**, incremental **+$17.600m**. These are verified *arithmetic identities given hypothetical assumptions*, not empirically verified exit values.

C8 uses IT Engineering **$25.5m M24 success EBITDA** in Base, whereas C9 uses **$24m transaction-pricing-date EBITDA**. The simulation selects **one profile explicitly**, never overlays C9's later stressed scenario table onto C8's execution table. When C9 company forecasts or programme uplift controls are edited while C8 is selected, their **changes relative to original C9 inputs** are shown as overlays on every C8 scenario; C8's state-specific starting assumptions remain intact.

Forecast revenues are **not supplied by C9**; their initially displayed values are placeholders initialized to each company's *latest historical revenue*. Changing them has no effect in the default **Direct EBITDA** forecast mode. Selecting **Revenue × Margin** activates the explicit formula.

## Financial model

For positive EBITDA:

```text
Enterprise value = EBITDA × EV/EBITDA
Equity value = max(0, enterprise value − exit net debt)
Fund gross receipts = equity value × effective ownership + entered contractual cash
M24 realized receipts = gross receipts × (1 − fees) × (1 − escrow − deferred)
                        if closed by M24, otherwise 0
Retained cash = budget − actual fund capital consumed
M24 terminal fund cash = Σ realized receipts + retained cash
Paired incremental = strategy M24 cash − matched No-Action M24 cash
Return generated per $1 of new fund capital = paired incremental ÷ capital consumed
```

When EBITDA is zero or negative, the model does **not** claim EV/EBITDA is reliable. It uses *only an explicit user-entered distressed equity recovery*; default is zero. The normal model assumes unchanged equity stakes, no priced equity or shareholder loans, no default fees, and M24 closures. Fees, escrow, deferred consideration, closing month, M24 closing probability and contractual cash can be entered, but these terms are not known from the case. Future NAV is excluded from **M24 cash**.

Fund capital is subtracted **once**, as part of retained capital; never a second time from exit receipts. The full programme must be financed by the fund and/or **explicitly confirmed** shareholders for the modeled effect to be available. Co-funding contributions, if any, are accounted for separately from fund capital.

The app includes a **Fund only**, **Pro-rata shareholder co-funding** and **Confirmed co-funding** mode. Shareholder funding is **not assumed as a fact**; the confirmation control must be checked for those illustrative modes. Priced equity and shareholder loans are disabled without the missing financial/legal terms. Staged alternatives invest a configurable initial fraction, with further capital paid only if the hypothetical milestone gate passes; if it does not, the full intervention uplift is **not** booked. No additional new investment returns are promised merely because partial funding is spent.

## Why the simulations are paired

C8 uses a common draw of the **Adverse, Base or Favorable** state. The market affects every company, and the **conditional execution state is sampled using the market-specific probabilities**. The default execution draw is shared across programmes, as in C8's conservative joint-outcome sensitivity; users may lower the shared probability for an illustrative idiosyncratic execution component.

The optional within-market EBITDA, multiple and net-debt shocks use standard normals truncated at ±3 standard deviations. A common-factor decomposition

```text
z_i = sqrt(rho) × common_factor + sqrt(1−rho) × company_factor_i
```

has a **positive-semidefinite** correlation structure for `rho ∈ [0, 1]`. Continuous shocks and exit-closing uniforms are reused across strategies and the matched No-Action calculation. Defaults for within-state volatility are zero: this reproduces the discrete C8 scenario arithmetic without adding unstated forecast dispersion. Changing financial assumptions changes every affected strategy dynamically.

Simulation outputs include expected and median M24 cash, 5th/10th/90th/95th cash percentiles, expected and median paired incremental cash, positive and negative paired-delta frequencies, 10% lower-tail CVaR, probabilities of completing all exits, and deployed-capital metrics. Probability estimates are **conditional on synthetic inputs**, not historically calibrated forecasts.

**C8 reconciliation reference** under 20% / 60% / 20% market weights and the source's conditional execution table (all exits close, no additional volatility):

| Strategy | Expected M24 cash | Expected incremental |
| --- | ---: | ---: |
| No-Action | 136.788 | 0.000 |
| IT only | 143.494 | +6.706 |
| Security Pilot | 138.247 | +1.459 |
| IT + Pilot | 144.953 | +8.165 |
| Security Full | 131.903 | −4.885 |
| E-commerce Full | 136.708 | −0.080 |
| IT + Pilot + E-commerce Partial | 145.758 | +8.970 |

Monte Carlo means converge to these expectation targets as the number of draws grows, with ordinary sampling variation.

## Controls and visualizations

- **Sidebar:** company No-Action EBITDA or revenue×margin, exit multiples, debt, ownership, distress recovery; programme costs, funded EBITDA improvement, multiple premiums, explicit funded debt changes; financing, co-funding and reserve; settlement and close timing; market/execution probabilities, volatilities and common-factor exposure.
- **Overview:** six cash/return KPIs, two strategy cash waterfalls, company cash breakdown, eligible strategy comparison, editable ranges for IT/Security/E-commerce heatmaps with break-even boundary, tornado impact chart.
- **Simulation:** distribution of strategy vs matched No-Action cash, paired-delta histogram, multi-strategy empirical CDFs, mean/P10/CVaR downside comparison, risk–return bubble plot and outperform probability chart.
- **Allocation:** current eligible discrete strategies, management priorities, capital feasibility, selectable deterministic/expected/probability/downside objectives, capital-to-expected-return Pareto frontier and side-by-side company cash waterfalls.
- **Audit/export:** source pages, formulas, current settings, seed, JSON scenario export/import, CSV exports and optional sensitivity chart PNG (when compatible Plotly/Kaleido is installed).

The app shows **STALE SIMULATION** when risk or valuation assumptions change after the last run. Re-run before interpreting or downloading those Monte Carlo results.

## Automated tests

```bash
python -m pytest -q
```

The test suite includes all **10** required C9 strategy cash fixtures, the mandatory No-Action and IT+Pilot identities, capital conservation, wasted intervention, paired zero difference, invalid probabilities, fixed-seed reproducibility, M24 exit exclusion, management/funding restrictions, monotonic valuation, forced 100% execution failure, and C8 large-sample scenario convergence. It also tests distress recovery, fee/escrow exclusions, co-funding gating, direct-vs-driver forecast modes, and milestone release.

To change C8/C9 defaults, edit the JSON files; keep analyst assumptions separate from historical facts and re-run tests.

## GitHub and Streamlit Community Cloud deployment

1. Push the **entire directory** to your GitHub repository (including `app.py`, `requirements.txt`, and `data/*.json`; include the original case PDF only if redistribution is permitted).
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, select the repository and branch.
3. Specify **Main file path:** `app.py`; select Python **3.11** if an interpreter setting is offered.
4. Deploy. There are **no secrets**, credentials, paid services, database, or backend processes. First deployment installs the packages from `requirements.txt`.
5. Validate the 10 C9 fixtures with `pytest` locally before presenting.

## Important limitations

- No buyer bids, verified seller cash waterfalls, current market quotes, legally committed shareholder contributions, ownership dilution terms, shareholder-loan recovery terms, bankruptcy claims or probabilities can be established from this PDF alone.
- Founder approval, operational implementation and mandatory M24 exits remain execution requirements; the app models potential sale noncompletion but **cannot guarantee** a buyer or founder agreement.
- Staged release gates, volatilities, costs, and co-funding assumptions are illustrative and **not** calibrated underwriting estimates.
- Operating or debt improvements are modeled only through explicit intervention assumptions. Co-funding cannot be treated as automatic; stages cannot be treated as fully successful when milestones fail.
- The optimization ranks **discrete permitted strategies** conditional on the selected objective, with no hard-coded company winner.
- Plotly chart-image export may require a compatible Kaleido binary. Chart display and CSV/JSON exports work independently.

## Source files

```text
app.py                  Streamlit UI and state management
model.py                Pure valuations, fees/settlement, cash reconciliation
simulation.py           Paired NumPy Monte Carlo engine
strategies.py           Strategy registry, financing and eligibility constraints
visualizations.py       Plotly interactive figures and heatmaps
data/*.json             Case facts, independent C9 and C8 configurations
data/Case Challenge (10).pdf
tests/                  Deterministic and stochastic financial tests
requirements.txt
.streamlit/config.toml  App theme
```
