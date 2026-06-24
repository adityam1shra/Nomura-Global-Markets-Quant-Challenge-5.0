// TABLE OF CONTENTS
// -----------------
// 1  Headers & Constants
// 2  Utilities       — parsing, dcf, CSV I/O
// 3  Core Data Types — Node, MarketQuote, SwapSpec, ParsedInput
// 4  Interpolator Interface
// 5  LinearInterpolator  (+CubicSpline stub)
// 6  AveragedQuadraticInterpolator
// 7  YieldCurve
// 8  Instruments     — CashInstrument, SwapInstrument  (+FRA stub)
// 9  Bootstrapper    — pattern B: calibrate(YieldCurve&) mutates directly
// 10 NewSwap Pricer
// 11 main & Output Writer
// ============================================================

// ─── 1  Headers & Constants ─────────────────────────────────
#include <algorithm>    // std::max, std::min (NR step clamp)
#include <array>        // std::array for 4-column output rows
#include <cassert>      // assert() in RUN_TESTS block
#include <cmath>        // std::log, std::exp, std::abs
#include <filesystem>   // std::filesystem::current_path, absolute
#include <fstream>      // std::ifstream, std::ofstream
#include <iomanip>      // std::setprecision, std::fixed
#include <iostream>     // std::cout (RUN_TESTS only)
#include <memory>       // std::unique_ptr
#include <sstream>      // std::istringstream for CSV field splitting
#include <stdexcept>    // std::runtime_error, std::logic_error
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace curves {

constexpr double kNotional    = 100.0;
constexpr int    kDaysPerYear = 360;

// ─── 2  Utilities ───────────────────────────────────────────

// Converts maturity token (e.g. "2W", "25Y") to integer days.
// D → ×1, W → ×7, M → ×30, Y → ×360.
int parseMaturityToDays(std::string_view tok) {
    if (tok.empty()) throw std::runtime_error("empty maturity token");
    char unit = static_cast<char>(std::toupper(static_cast<unsigned char>(tok.back())));
    std::string numStr(tok.substr(0, tok.size() - 1));
    if (numStr.empty())
        throw std::runtime_error("maturity token missing number: " + std::string(tok));
    int n = std::stoi(numStr);
    switch (unit) {
        case 'D': return n;
        case 'W': return n * 7;
        case 'M': return n * 30;
        case 'Y': return n * kDaysPerYear;
        default:
            throw std::runtime_error(
                "unknown maturity unit '" + std::string(1, unit) + "' in: " + std::string(tok));
    }
}

// Converts frequency token to integer days. Only 1m/3m/6m/12m accepted.
int parseFrequencyToDays(std::string_view tok) {
    if (tok == "1m")  return 30;
    if (tok == "3m")  return 90;
    if (tok == "6m")  return 180;
    if (tok == "12m") return 360;
    throw std::runtime_error("unknown frequency token: " + std::string(tok));
}

// Day-count fraction: (to_days - from_days) / 360.
double dcf(int from_days, int to_days) {
    return static_cast<double>(to_days - from_days) / static_cast<double>(kDaysPerYear);
}

// Strips a UTF-8 BOM (EF BB BF) if present at start of stream.
void stripBOM(std::istream& in) {
    if (static_cast<unsigned char>(in.peek()) != 0xEF) return;
    char bom[3];
    in.read(bom, 3);
    if (static_cast<unsigned char>(bom[0]) != 0xEF ||
        static_cast<unsigned char>(bom[1]) != 0xBB ||
        static_cast<unsigned char>(bom[2]) != 0xBF)
        in.seekg(0);
}

// Reads one logical line normalising CRLF → LF. Returns false at EOF.
bool readLine(std::istream& in, std::string& out) {
    if (!std::getline(in, out)) return false;
    if (!out.empty() && out.back() == '\r') out.pop_back();
    return true;
}

// Splits a CSV line into whitespace-trimmed fields.
std::vector<std::string> splitCSV(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        auto s = field.find_first_not_of(" \t\r\n");
        auto e = field.find_last_not_of(" \t\r\n");
        fields.push_back(s == std::string::npos ? "" : field.substr(s, e - s + 1));
    }
    return fields;
}

// ─── 3  Core Data Types ──────────────────────────────────────

struct Node {
    int    days;  // maturity in days from today (t=0)
    double lnDF;  // ln(discount factor) at this maturity
};

struct MarketQuote {
    int    days;        // maturity in days
    double cashRate;    // decimal (converted from % at parse time)
    double parSwapRate; // decimal
};

struct SwapSpec {
    double notional;
    double fixedRate;     // decimal
    int    maturityDays;
    int    fixedFreqDays;
    int    floatFreqDays;
};

struct ParsedInput {
    int                      N;
    std::vector<MarketQuote> quotes;
    int                      lookupDays;
    SwapSpec                 newSwap;
};

// ─── 2 (cont.)  CSV Parser ──────────────────────────────────

ParsedInput parseInput(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("cannot open: " + path);
    stripBOM(file);

    ParsedInput result{};
    std::string line;

    // Row 1: N
    if (!readLine(file, line)) throw std::runtime_error("missing row 1 (N)");
    {
        auto f = splitCSV(line);
        if (f.empty() || f[0].empty()) throw std::runtime_error("row 1: missing N");
        result.N = std::stoi(f[0]);
        if (result.N < 2)
            throw std::runtime_error("N must be >= 2, got " + std::to_string(result.N));
    }

    // Rows 2..N+1: Maturity, CashRate%, ParSwapRate%
    result.quotes.reserve(static_cast<std::size_t>(result.N));
    for (int i = 0; i < result.N; ++i) {
        int rowNum = i + 2;
        if (!readLine(file, line))
            throw std::runtime_error("missing quote row " + std::to_string(rowNum));
        auto f = splitCSV(line);
        if (f.size() < 3)
            throw std::runtime_error("quote row " + std::to_string(rowNum) + ": need 3 fields");

        MarketQuote q{};
        q.days = parseMaturityToDays(f[0]);

        if (f[1].empty())
            throw std::runtime_error("quote row " + std::to_string(rowNum) + ": missing cashRate");
        if (f[2].empty())
            throw std::runtime_error("quote row " + std::to_string(rowNum) + ": missing parSwapRate");

        q.cashRate    = std::stod(f[1]) / 100.0;
        q.parSwapRate = std::stod(f[2]) / 100.0;

        if (q.cashRate == 0.0)
            throw std::runtime_error("quote row " + std::to_string(rowNum) + ": cashRate is zero");
        if (q.parSwapRate == 0.0)
            throw std::runtime_error("quote row " + std::to_string(rowNum) + ": parSwapRate is zero");

        result.quotes.push_back(q);
    }

    // Row N+2: lookup day t
    if (!readLine(file, line)) throw std::runtime_error("missing lookup-day row");
    {
        auto f = splitCSV(line);
        if (f.empty() || f[0].empty()) throw std::runtime_error("lookup-day row: missing value");
        result.lookupDays = std::stoi(f[0]);
        if (result.lookupDays < 0)
            throw std::runtime_error("lookup day must be >= 0, got " +
                                     std::to_string(result.lookupDays));
    }

    // Row N+3: FixedRate%, Maturity, FixedFreq, FloatFreq
    if (!readLine(file, line)) throw std::runtime_error("missing swap-spec row");
    {
        auto f = splitCSV(line);
        if (f.size() < 4) throw std::runtime_error("swap-spec row: need 4 fields");

        SwapSpec s{};
        s.notional      = kNotional;
        s.fixedRate     = std::stod(f[0]) / 100.0;
        s.maturityDays  = parseMaturityToDays(f[1]);
        s.fixedFreqDays = parseFrequencyToDays(f[2]);
        s.floatFreqDays = parseFrequencyToDays(f[3]);

        if (s.maturityDays % s.fixedFreqDays != 0)
            throw std::runtime_error("swap maturity not divisible by fixed frequency");
        if (s.maturityDays % s.floatFreqDays != 0)
            throw std::runtime_error("swap maturity not divisible by float frequency");

        result.newSwap = s;
    }

    return result;
}

// ─── 4  Interpolator Interface ──────────────────────────────

class Interpolator {
public:
    virtual ~Interpolator() = default;

    // Returns ln(DF(t)) by interpolating over sorted nodes.
    // Pre-condition: T_0 <= t <= T_{n-1}, n >= 1.
    virtual double logDF(int t, const std::vector<Node>& nodes) const = 0;

    // Returns ∂ln(DF(t))/∂ln(DF(T_k)) for every node k.
    // Weights sum to 1 and are non-negative for convex interpolators.
    virtual std::vector<double> nodeWeights(int t, const std::vector<Node>& nodes) const = 0;
};

// ─── 5  LinearInterpolator  (+CubicSpline stub) ─────────────

class LinearInterpolator : public Interpolator {
    // Finds index i s.t. nodes[i].days <= t <= nodes[i+1].days.
    // Clamps to last valid interval if t == nodes.back().days.
    static std::size_t findInterval(int t, const std::vector<Node>& nodes) {
        std::size_t n = nodes.size();
        std::size_t i = 0;
        for (std::size_t k = 0; k < n; ++k) {
            if (nodes[k].days <= t) i = k;
            else break;
        }
        if (i == n - 1) i = n - 2; // clamp: t == last node
        return i;
    }

public:
    double logDF(int t, const std::vector<Node>& nodes) const override {
        if (nodes.size() == 1) return nodes[0].lnDF;
        std::size_t i = findInterval(t, nodes);
        double T0    = static_cast<double>(nodes[i].days);
        double T1    = static_cast<double>(nodes[i + 1].days);
        double alpha = (static_cast<double>(t) - T0) / (T1 - T0);
        return (1.0 - alpha) * nodes[i].lnDF + alpha * nodes[i + 1].lnDF;
    }

    std::vector<double> nodeWeights(int t, const std::vector<Node>& nodes) const override {
        std::size_t n = nodes.size();
        std::vector<double> w(n, 0.0);
        if (n == 1) { w[0] = 1.0; return w; }
        std::size_t i = findInterval(t, nodes);
        double T0    = static_cast<double>(nodes[i].days);
        double T1    = static_cast<double>(nodes[i + 1].days);
        double alpha = (static_cast<double>(t) - T0) / (T1 - T0);
        w[i]     = 1.0 - alpha;
        w[i + 1] = alpha;
        return w;
    }
};

// Extension point — see documentation.docx
class CubicSplineInterpolator : public Interpolator {
public:
    double logDF(int, const std::vector<Node>&) const override {
        throw std::logic_error("CubicSpline not implemented — extension point for Q3");
    }
    std::vector<double> nodeWeights(int, const std::vector<Node>&) const override {
        throw std::logic_error("CubicSpline not implemented — extension point for Q3");
    }
};

// ─── 6  AveragedQuadraticInterpolator ───────────────────────
//
// For t ∈ [T_i, T_{i+1}], α = (t−T_i)/(T_{i+1}−T_i):
//   first interval  (i == 0):    log-linear fallback — no left neighbour
//   last interval   (i == n−2):  single Lagrange parabola through (n−3, n−2, n−1)
//   interior        (1 ≤ i ≤ n−3): blend (1−α)·q1 + α·q2
//     q1 = Lagrange parabola through nodes (i−1, i, i+1)
//     q2 = Lagrange parabola through nodes (i, i+1, i+2)
// logDF delegates to nodeWeights so the two are always consistent.

class AveragedQuadraticInterpolator : public Interpolator {
    // Same clamping rule as LinearInterpolator.
    static std::size_t findInterval(int t, const std::vector<Node>& nodes) {
        std::size_t n = nodes.size();
        std::size_t i = 0;
        for (std::size_t k = 0; k < n; ++k) {
            if (nodes[k].days <= t) i = k;
            else break;
        }
        if (i == n - 1) i = n - 2;
        return i;
    }

    // Lagrange basis weights [wa, wb, wc] for scalar t through knots (Ta, Tb, Tc).
    // May produce negative weights — that is expected for higher-order interpolation.
    static std::array<double, 3> lagrange3(double t, double Ta, double Tb, double Tc) {
        double wa = (t - Tb) * (t - Tc) / ((Ta - Tb) * (Ta - Tc));
        double wb = (t - Ta) * (t - Tc) / ((Tb - Ta) * (Tb - Tc));
        double wc = (t - Ta) * (t - Tb) / ((Tc - Ta) * (Tc - Tb));
        return {wa, wb, wc};
    }

public:
    // Implemented as dot(nodeWeights, lnDF) to guarantee exact consistency.
    double logDF(int t, const std::vector<Node>& nodes) const override {
        auto w = nodeWeights(t, nodes);
        double v = 0.0;
        for (std::size_t k = 0; k < nodes.size(); ++k) v += w[k] * nodes[k].lnDF;
        return v;
    }

    std::vector<double> nodeWeights(int t, const std::vector<Node>& nodes) const override {
        std::size_t n = nodes.size();
        std::vector<double> w(n, 0.0);
        if (n == 1) { w[0] = 1.0; return w; }

        std::size_t i = findInterval(t, nodes);
        double dt  = static_cast<double>(t);
        double Ti  = static_cast<double>(nodes[i].days);
        double Ti1 = static_cast<double>(nodes[i + 1].days);
        double alpha = (dt - Ti) / (Ti1 - Ti);

        if (i == 0) {
            // First interval: no i−1 exists — degrade to log-linear.
            w[0] = 1.0 - alpha;
            w[1] = alpha;
            return w;
        }

        if (i == n - 2) {
            // Last interval: no i+2 exists — single parabola through (n−3, n−2, n−1).
            auto lw = lagrange3(dt,
                static_cast<double>(nodes[n - 3].days),
                static_cast<double>(nodes[n - 2].days),
                static_cast<double>(nodes[n - 1].days));
            w[n - 3] = lw[0];
            w[n - 2] = lw[1];
            w[n - 1] = lw[2];
            return w;
        }

        // Interior interval: blend q1=(i−1,i,i+1) and q2=(i,i+1,i+2).
        double Ti_1 = static_cast<double>(nodes[i - 1].days);
        double Ti2  = static_cast<double>(nodes[i + 2].days);

        auto q1 = lagrange3(dt, Ti_1, Ti,  Ti1);  // for nodes i−1, i, i+1
        auto q2 = lagrange3(dt, Ti,   Ti1, Ti2);  // for nodes i, i+1, i+2

        w[i - 1] = (1.0 - alpha) * q1[0];
        w[i]     = (1.0 - alpha) * q1[1] + alpha * q2[0];
        w[i + 1] = (1.0 - alpha) * q1[2] + alpha * q2[1];
        w[i + 2] =         alpha * q2[2];
        return w;
    }
};

// ─── 7  YieldCurve ──────────────────────────────────────────

class YieldCurve {
    std::vector<Node>                nodes_;
    std::unique_ptr<Interpolator>    interp_;
    std::vector<std::vector<double>> jacobian_; // [k][j] = ∂lnDF(T_k)/∂quote_j

public:
    explicit YieldCurve(std::unique_ptr<Interpolator> interp)
        : interp_(std::move(interp)) {}

    // Appends a calibrated node. Caller is responsible for sorted order.
    void addNode(int days, double lnDF) {
        nodes_.push_back({days, lnDF});
    }

    // Overwrites the lnDF of the last node in place.
    // Used by SwapInstrument to update the trial node each Newton-Raphson iteration.
    void updateLastNodeLnDF(double lnDF) {
        if (nodes_.empty()) throw std::runtime_error("updateLastNodeLnDF: no nodes");
        nodes_.back().lnDF = lnDF;
    }

    // Stores the Jacobian row for node k. Called by Bootstrapper.
    void setJacobianRow(std::size_t k, std::vector<double> row) {
        if (jacobian_.size() <= k) jacobian_.resize(k + 1);
        jacobian_[k] = std::move(row);
    }

    const std::vector<Node>&                nodes()        const { return nodes_; }
    const std::vector<std::vector<double>>& nodeJacobian() const { return jacobian_; }

    // Returns ln(DF(t)). Enforces extrapolation policy:
    //   t < 0           → throw
    //   t == 0          → 0.0  (ln(1))
    //   0 < t < T_0     → throw  (FlatExtrapolation is a future extension point)
    //   T_0 ≤ t ≤ T_n-1 → interpolate
    //   t > T_n-1       → throw
    double logDF(int t) const {
        if (t < 0)
            throw std::runtime_error("logDF: t=" + std::to_string(t) + " < 0");
        if (t == 0) return 0.0;
        if (nodes_.empty())
            throw std::runtime_error("logDF: curve has no nodes");
        if (t < nodes_.front().days)
            throw std::runtime_error("logDF: t=" + std::to_string(t) +
                                     " < first node T=" + std::to_string(nodes_.front().days) +
                                     " (left extrapolation not supported)");
        if (t > nodes_.back().days)
            throw std::runtime_error("logDF: t=" + std::to_string(t) +
                                     " > last node T=" + std::to_string(nodes_.back().days) +
                                     " (right extrapolation not supported)");
        return interp_->logDF(t, nodes_);
    }

    // Returns DF(t) = exp(logDF(t)). Same extrapolation policy; t==0 returns 1.0 exactly.
    double DF(int t) const {
        if (t == 0) return 1.0;
        return std::exp(logDF(t));
    }

    // Returns ∂ln(DF(t))/∂ln(DF(T_k)) for every node k.
    // At t==0 returns zero vector (DF(0)=1 regardless of curve shape).
    std::vector<double> nodeWeights(int t) const {
        if (t == 0) return std::vector<double>(nodes_.size(), 0.0);
        // logDF enforces the range check; call it first to trigger any throws.
        logDF(t);
        return interp_->nodeWeights(t, nodes_);
    }
};

// ─── 8  Instruments ─────────────────────────────────────────

class Instrument {
public:
    virtual ~Instrument() = default;
    virtual void calibrate(YieldCurve& curve, double quote)          = 0;
    virtual double presentValue(const YieldCurve& curve)       const = 0;
    // Returns ∂PV/∂ln(DF(T_k)) for every node k.
    virtual std::vector<double> sensitivities(const YieldCurve&) const = 0;
};

// Cash deposit: DF(T) = 1 / (1 + cashRate * T/360).
class CashInstrument : public Instrument {
    int days_;
public:
    explicit CashInstrument(int days) : days_(days) {}

    void calibrate(YieldCurve& curve, double cashRate) override {
        double tau  = dcf(0, days_);
        double lnDF = -std::log(1.0 + cashRate * tau);
        std::size_t k = curve.nodes().size(); // index of the node about to be added
        curve.addNode(days_, lnDF);

        // Diagonal Jacobian: ∂lnDF(T_k)/∂cashRate_k = -tau / (1 + cashRate*tau)
        // All off-diagonal entries are 0 (cash nodes are independent).
        std::vector<double> row(k + 1, 0.0);
        row[k] = -tau / (1.0 + cashRate * tau);
        curve.setJacobianRow(k, std::move(row));
    }

    // Cash instruments are calibration-only in this design; pricing goes through
    // NewSwap / run(). Calling these methods indicates a design error.
    double presentValue(const YieldCurve&) const override {
        throw std::logic_error("not used by this submission");
    }
    std::vector<double> sensitivities(const YieldCurve&) const override {
        throw std::logic_error("not used by this submission");
    }
};

// Par-swap quoted instrument.
// T ≤ 180 d: simple cash-deposit formula on the par rate.
// T > 180 d: semi-annual payment schedule; Newton-Raphson on lnDF(T_new) so that
//             the computed par rate (using the interpolator for mid-period dates)
//             matches the market quote.
class SwapInstrument : public Instrument {
    int    days_;
    double parSwapRate_ = 0.0; // stored at calibrate() for presentValue/sensitivities

    // Build the sorted list of semi-annual payment dates for this swap.
    std::vector<int> paymentDates() const {
        std::vector<int> dates;
        for (int t = 180; t <= days_; t += 180) dates.push_back(t);
        return dates;
    }

public:
    explicit SwapInstrument(int days) : days_(days) {}

    void calibrate(YieldCurve& curve, double parSwapRate) override {
        parSwapRate_ = parSwapRate;
        if (days_ <= 180) {
            // Short maturity: closed-form, identical to CashInstrument logic.
            double tau  = dcf(0, days_);
            double lnDF = -std::log(1.0 + parSwapRate * tau);
            std::size_t k = curve.nodes().size();
            curve.addNode(days_, lnDF);
            std::vector<double> row(k + 1, 0.0);
            row[k] = -tau / (1.0 + parSwapRate * tau); // ∂lnDF/∂p (diagonal)
            curve.setJacobianRow(k, std::move(row));
            return;
        }

        // Long maturity: Newton-Raphson on lnDF(T_new) — no solver abstraction.
        // Objective: F(x) = computedParRate(x) - marketParRate = 0,  x = lnDF(T_new)
        // Derivative: ∂F/∂x = ∂parRate/∂x via quotient rule + interpolator nodeWeights.
        auto dates       = paymentDates();
        std::size_t kNew = curve.nodes().size();

        // Capture previous node before adding the new placeholder.
        double lnDFprev = curve.nodes().empty() ? 0.0 : curve.nodes().back().lnDF;
        int    daysPrev = curve.nodes().empty() ? 0    : curve.nodes().back().days;

        // Initial guess: lnDF_T ≈ lnDF_prev − marketRate · (T − T_prev)/360
        double lnDF = lnDFprev
                    - parSwapRate * static_cast<double>(days_ - daysPrev) / 360.0;
        curve.addNode(days_, lnDF); // placeholder — mutated in-place each NR step

        constexpr int    kNRMax   = 50;
        constexpr double kNRTol   = 1e-12;
        constexpr double kNRClamp = 1.0;   // max |Δlndf| per step

        double dF_dlnDF  = 0.0;
        bool   converged = false;

        for (int it = 0; it < kNRMax; ++it) {
            curve.updateLastNodeLnDF(lnDF);
            double dfT = std::exp(lnDF);

            // Annuity D = Σ_j DF(t_j)·0.5  and  ∂D/∂lnDF_T
            // ∂DF(t_j)/∂lnDF_T = DF(t_j) · nodeWeights(t_j)[kNew]
            double D = 0.0, dD = 0.0;
            for (int ti : dates) {
                double dfi  = curve.DF(ti);
                double w_kT = curve.nodeWeights(ti)[kNew]; // ∂lnDF(ti)/∂lnDF_T
                D  += 0.5 * dfi;
                dD += 0.5 * dfi * w_kT;
            }

            double parRate = (1.0 - dfT) / D;
            double F       = parRate - parSwapRate;

            // Quotient rule: N=1−dfT, N'=−dfT, D'=dD
            // ∂parRate/∂lnDF_T = (N'D − N·D') / D²
            dF_dlnDF = (-dfT * D - (1.0 - dfT) * dD) / (D * D);

            if (std::abs(F) < kNRTol) { converged = true; break; }

            if (!std::isfinite(dF_dlnDF) || dF_dlnDF == 0.0)
                throw std::runtime_error("NR: degenerate derivative at T=" +
                                          std::to_string(days_));

            double step = F / dF_dlnDF;
            step = std::max(-kNRClamp, std::min(kNRClamp, step)); // clamp ±1.0
            lnDF -= step;

            if (!std::isfinite(lnDF))
                throw std::runtime_error("NR: diverged at T=" + std::to_string(days_) +
                                          " it=" + std::to_string(it));
        }
        if (!converged)
            throw std::runtime_error("NR: max iter=" + std::to_string(kNRMax) +
                                      " reached for T=" + std::to_string(days_));

        curve.updateLastNodeLnDF(lnDF);

        // ── IFT lower-triangular Jacobian row ────────────────────────────────────
        //
        // F_k = parRate - p_k = 0 at convergence (p_k is the market par swap rate).
        // S_k := ∂F_k/∂lnDF(T_k) = dF_dlnDF  (stored above, < 0)
        //
        // By IFT:
        //   J[k][k] = ∂lnDF(T_k)/∂p_k = 1/S_k                           (diagonal)
        //   J[k][j] = -(1/S_k) · Σ_{m<k} (∂F_k/∂lnDF(T_m)) · J[m][j]  (j < k)
        //
        // ∂F_k/∂lnDF(T_m) = -(1-dfT)/D² · dD_m  where dD_m = ∂D/∂lnDF(T_m)
        //                  = Σ_{ti∈dates} 0.5 · DF(ti) · nodeWeights(ti)[m]
        //
        // Note: nodeWeights depend only on grid positions, not on lnDF values,
        // so calling curve.nodeWeights(ti) is safe here.

        // Recompute D and dfT at convergence (curve already updated above).
        double dfT_conv = std::exp(lnDF);
        double D_conv   = 0.0;
        for (int ti : dates) D_conv += 0.5 * curve.DF(ti);

        // Step 1: dD_m = ∂D/∂lnDF(T_m) for each prior node m = 0..kNew-1
        std::vector<double> dD_m(kNew, 0.0);
        for (int ti : dates) {
            double df_ti = curve.DF(ti);
            auto   wts   = curve.nodeWeights(ti);   // length = kNew+1
            for (std::size_t m = 0; m < kNew; ++m)
                dD_m[m] += 0.5 * df_ti * wts[m];
        }

        // Step 2: precompute the common multiplier
        // ∂F_k/∂lnDF(T_m) = coeff · dD_m[m]  where coeff = -(1-dfT)/D² < 0
        double coeff  = -(1.0 - dfT_conv) / (D_conv * D_conv);
        double inv_Sk = 1.0 / dF_dlnDF;   // < 0

        // Step 3: build the Jacobian row
        std::vector<double> row(kNew + 1, 0.0);
        row[kNew] = inv_Sk;   // diagonal: 1/S_k < 0

        // Read prior Jacobian rows before potentially resizing them.
        {
            const auto& jac = curve.nodeJacobian();
            for (std::size_t j = 0; j < kNew; ++j) {
                double sum = 0.0;
                for (std::size_t m = 0; m < kNew; ++m) {
                    if (m >= jac.size()) break;
                    // J[m][j] = 0 for j > m (lower triangular; jac[m] has size m+1)
                    double J_mj = (j < jac[m].size()) ? jac[m][j] : 0.0;
                    sum += coeff * dD_m[m] * J_mj;  // Σ (∂F_k/∂lnDF_m) · J[m][j]
                }
                row[j] = -inv_Sk * sum;  // -(1/S_k) · Σ (∂F_k/∂lnDF_m) · J[m][j]
            }
        } // jac reference dropped before setJacobianRow (avoids invalidation risk)
        curve.setJacobianRow(kNew, std::move(row));
    }

    // PV = (1 − DF(T)) − p · Σ_j DF(t_j) · DCF_j   (unit notional, fixed-payer)
    // At calibration this evaluates to zero by construction.
    double presentValue(const YieldCurve& curve) const override {
        auto dates = paymentDates();
        double dfT    = curve.DF(days_);
        double annuity = 0.0;
        int prev = 0;
        for (int ti : dates) {
            annuity += curve.DF(ti) * dcf(prev, ti);
            prev = ti;
        }
        return (1.0 - dfT) - parSwapRate_ * annuity;
    }

    // Returns ∂PV/∂lnDF(T_k) for every curve node k  (unit notional).
    // This is the raw curve sensitivity before any quote-Jacobian multiplication.
    std::vector<double> sensitivities(const YieldCurve& curve) const override {
        const std::size_t nNodes = curve.nodes().size();
        auto dates = paymentDates();
        std::vector<double> sens(nNodes, 0.0);
        int prev = 0;
        for (int tj : dates) {
            double dcf_j     = dcf(prev, tj);  prev = tj;
            double dPV_dDF_j = -parSwapRate_ * dcf_j;
            if (tj == days_) dPV_dDF_j -= 1.0;  // notional = 1
            double contrib = curve.DF(tj) * dPV_dDF_j;
            auto   w       = curve.nodeWeights(tj);
            for (std::size_t k = 0; k < nNodes; ++k)
                sens[k] += contrib * w[k];
        }
        return sens;
    }
};

// Extension point — see documentation.docx
class ForwardRateAgreement : public Instrument {
public:
    void calibrate(YieldCurve&, double) override {
        throw std::logic_error("ForwardRateAgreement not implemented — extension point for Q3");
    }
    double presentValue(const YieldCurve&) const override {
        throw std::logic_error("ForwardRateAgreement not implemented — extension point for Q3");
    }
    std::vector<double> sensitivities(const YieldCurve&) const override {
        throw std::logic_error("ForwardRateAgreement not implemented — extension point for Q3");
    }
};

// ─── 9  Bootstrapper ────────────────────────────────────────
// Pattern B: instrument.calibrate(YieldCurve&, quote) mutates the curve directly.

YieldCurve buildCashCurve(const std::vector<MarketQuote>& quotes,
                           std::unique_ptr<Interpolator>   interp) {
    YieldCurve curve(std::move(interp));
    for (const auto& q : quotes) {
        CashInstrument instr(q.days);
        instr.calibrate(curve, q.cashRate);
    }
    return curve;
}

YieldCurve buildSwapCurve(const std::vector<MarketQuote>& quotes,
                           std::unique_ptr<Interpolator>   interp) {
    YieldCurve curve(std::move(interp));
    for (const auto& q : quotes) {
        SwapInstrument instr(q.days);
        instr.calibrate(curve, q.parSwapRate);
    }
    return curve;
}

// ─── 10  NewSwap Pricer ─────────────────────────────────────
//
// Fixed-payer swap: pay fixed K, receive floating.
// Float leg telescopes regardless of float frequency:
//   PV_float = N · (1 − DF(T_final))
// Fixed leg:
//   PV_fixed = N · K · Σ_i DF(t_i^fix) · DCF_i
// Net PV (fixed payer) = PV_float − PV_fixed
// Par rate K* : set PV=0  →  K* = (1−DF(T_final)) / Σ_i DF(t_i^fix)·DCF_i

class NewSwap {
    SwapSpec spec_;

    // Fixed-leg payment dates: fixedFreq, 2·fixedFreq, …, maturity
    std::vector<int> fixedDates() const {
        std::vector<int> v;
        for (int t = spec_.fixedFreqDays; t <= spec_.maturityDays; t += spec_.fixedFreqDays)
            v.push_back(t);
        return v;
    }

    // Shared helper: returns {dfFinal, annuity}.
    std::pair<double, double> legComponents(const YieldCurve& curve) const {
        double dfFinal = curve.DF(spec_.maturityDays);
        double annuity = 0.0;
        int prev = 0;
        for (int t : fixedDates()) {
            annuity += curve.DF(t) * dcf(prev, t);
            prev = t;
        }
        return {dfFinal, annuity};
    }

public:
    explicit NewSwap(const SwapSpec& s) : spec_(s) {}

    // PV = N·(1−DF(T)) − N·K·annuity
    double PV(const YieldCurve& curve) const {
        auto [dfFinal, annuity] = legComponents(curve);
        return spec_.notional * ((1.0 - dfFinal) - spec_.fixedRate * annuity);
    }

    // Par rate K* = (1−DF(T)) / annuity
    double parRate(const YieldCurve& curve) const {
        auto [dfFinal, annuity] = legComponents(curve);
        return (1.0 - dfFinal) / annuity;
    }
};

// ─── 11  Output Writer ──────────────────────────────────────

void writeOutput(const std::string& path,
                 const std::vector<std::array<double, 4>>& rows) {
    std::ofstream out(path);
    if (!out.is_open()) throw std::runtime_error("cannot open for write: " + path);
    out << std::fixed << std::setprecision(10);
    for (const auto& row : rows)
        out << row[0] << ',' << row[1] << ',' << row[2] << ',' << row[3] << '\n';
}

// ─── 11  Engine entry-point ─────────────────────────────────

int run() {
    namespace fs = std::filesystem;

    const std::string outputName = "Output.csv";
    ParsedInput       data       = parseInput(fs::absolute("Input.csv").string());
    const auto&       s          = data.newSwap;

    // ── Build all four curves ────────────────────────────────────────────────
    YieldCurve cashLinear = buildCashCurve(data.quotes,
                                            std::make_unique<LinearInterpolator>());
    YieldCurve cashAQ     = buildCashCurve(data.quotes,
                                            std::make_unique<AveragedQuadraticInterpolator>());
    YieldCurve swapLinear = buildSwapCurve(data.quotes,
                                            std::make_unique<LinearInterpolator>());
    YieldCurve swapAQ     = buildSwapCurve(data.quotes,
                                            std::make_unique<AveragedQuadraticInterpolator>());

    // ── Row 1: DF(lookupDays) on all four curves ────────────────────────────
    double dfCL = cashLinear.DF(data.lookupDays);
    double dfCA = cashAQ    .DF(data.lookupDays);
    double dfSL = swapLinear.DF(data.lookupDays);
    double dfSA = swapAQ    .DF(data.lookupDays);

    // ── Rows 2–3: new-swap PV and par rate ──────────────────────────────────
    NewSwap swap(data.newSwap);
    double pvCL  = swap.PV(cashLinear);  double parCL  = swap.parRate(cashLinear);
    double pvCA  = swap.PV(cashAQ);      double parCA  = swap.parRate(cashAQ);
    double pvSL  = swap.PV(swapLinear);  double parSL  = swap.parRate(swapLinear);
    double pvSA  = swap.PV(swapAQ);      double parSA  = swap.parRate(swapAQ);

    // ── Assemble output row array (rows 1–3 fixed; rows 4–N+3 from risk) ────
    std::vector<std::array<double, 4>> outRows(
        static_cast<std::size_t>(3 + data.N),
        std::array<double, 4>{0.0, 0.0, 0.0, 0.0});
    outRows[0] = {dfCL,  dfCA,  dfSL,  dfSA };
    outRows[1] = {pvCL,  pvCA,  pvSL,  pvSA };
    outRows[2] = {parCL, parCA, parSL, parSA};

    // ── Cash-rate DV01: ∂PV/∂c_i ────────────────────────────────────────────
    // ∂PV/∂c_i = [Σ_j (∂PV/∂DF(t_j)) · DF(t_j) · w_i(t_j)] · J_diag[i]
    // J_diag[i] = -τ_i · DF(T_i)  (diagonal Jacobian stored by CashInstrument)
    auto computeCashRisk = [&](const YieldCurve& curve) -> std::vector<double> {
        const std::size_t nNodes = curve.nodes().size();
        std::vector<int> pmts;
        for (int t = s.fixedFreqDays; t <= s.maturityDays; t += s.fixedFreqDays)
            pmts.push_back(t);
        std::vector<double> dPV_dlnDF(nNodes, 0.0);
        int prev = 0;
        for (int tj : pmts) {
            double dcf_j     = dcf(prev, tj);  prev = tj;
            double dPV_dDF_j = -s.notional * s.fixedRate * dcf_j;
            if (tj == s.maturityDays) dPV_dDF_j -= s.notional;
            double contrib = curve.DF(tj) * dPV_dDF_j;
            auto   w       = curve.nodeWeights(tj);
            for (std::size_t i = 0; i < nNodes; ++i)
                dPV_dlnDF[i] += contrib * w[i];
        }
        const auto& jac = curve.nodeJacobian();
        std::vector<double> risk(nNodes, 0.0);
        for (std::size_t i = 0; i < nNodes; ++i) {
            double J_ii = (i < jac.size() && i < jac[i].size()) ? jac[i][i] : 0.0;
            risk[i] = dPV_dlnDF[i] * J_ii;
        }
        return risk;
    };
    auto riskCL = computeCashRisk(cashLinear);
    auto riskCA = computeCashRisk(cashAQ);
    for (std::size_t i = 0; i < static_cast<std::size_t>(data.N); ++i) {
        outRows[3 + i][0] = riskCL[i];
        outRows[3 + i][1] = riskCA[i];
    }

    // ── Par-swap-rate DV01: ∂PV/∂p_j ────────────────────────────────────────
    // risk[j] = Σ_k v[k] · J[k][j]  (J = lower-triangular IFT Jacobian)
    auto computeSwapRisk = [&](const YieldCurve& curve) -> std::vector<double> {
        const std::size_t nNodes = curve.nodes().size();
        std::vector<int> pmts;
        for (int t = s.fixedFreqDays; t <= s.maturityDays; t += s.fixedFreqDays)
            pmts.push_back(t);
        std::vector<double> v(nNodes, 0.0);
        int prev = 0;
        for (int tj : pmts) {
            double dcf_j     = dcf(prev, tj);  prev = tj;
            double dPV_dDF_j = -s.notional * s.fixedRate * dcf_j;
            if (tj == s.maturityDays) dPV_dDF_j -= s.notional;
            double contrib = curve.DF(tj) * dPV_dDF_j;
            auto   w       = curve.nodeWeights(tj);
            for (std::size_t k = 0; k < nNodes; ++k)
                v[k] += contrib * w[k];
        }
        const auto& jac = curve.nodeJacobian();
        std::vector<double> risk(nNodes, 0.0);
        for (std::size_t k = 0; k < nNodes; ++k) {
            if (k >= jac.size()) continue;
            for (std::size_t j = 0; j < jac[k].size(); ++j)
                risk[j] += v[k] * jac[k][j];
        }
        return risk;
    };
    auto riskSL = computeSwapRisk(swapLinear);
    auto riskSA = computeSwapRisk(swapAQ);
    for (std::size_t i = 0; i < static_cast<std::size_t>(data.N); ++i) {
        outRows[3 + i][2] = riskSL[i];
        outRows[3 + i][3] = riskSA[i];
    }

    writeOutput(outputName, outRows);
    return 0;
}

} // namespace curves

// ─── Global main ─────────────────────────────────────────────

int main() {
#ifdef RUN_TESTS
    using namespace curves;

    // ── Parse utilities ──────────────────────────────────────
    assert(parseMaturityToDays("1D")  == 1);
    assert(parseMaturityToDays("2W")  == 14);
    assert(parseMaturityToDays("3M")  == 90);
    assert(parseMaturityToDays("25Y") == 9000);

    assert(parseFrequencyToDays("1m")  == 30);
    assert(parseFrequencyToDays("6m")  == 180);

    assert(dcf(0, 360) == 1.0);
    assert(dcf(0, 180) == 0.5);

    auto mustThrow = [](auto fn) {
        bool threw = false;
        try { fn(); } catch (...) { threw = true; }
        assert(threw);
    };
    mustThrow([] { parseMaturityToDays("1X"); });
    mustThrow([] { parseMaturityToDays("");   });
    mustThrow([] { parseFrequencyToDays("4m"); });

    // ── LinearInterpolator ────────────────────────────────────
    {
        std::vector<Node> nodes = {
            {30,  std::log(0.99)},
            {90,  std::log(0.97)},
            {180, std::log(0.94)}
        };
        LinearInterpolator li;
        std::size_t n = nodes.size();

        // nodeWeights returns a unit vector at every node
        for (std::size_t k = 0; k < n; ++k) {
            auto w = li.nodeWeights(nodes[k].days, nodes);
            assert(w.size() == n);
            for (std::size_t j = 0; j < n; ++j) {
                if (j == k) assert(std::abs(w[j] - 1.0) < 1e-15);
                else        assert(std::abs(w[j])       < 1e-15);
            }
        }

        // weights sum to 1 at interior points
        for (int t : {45, 60, 120, 150}) {
            auto w = li.nodeWeights(t, nodes);
            double sum = 0.0;
            for (double x : w) sum += x;
            assert(std::abs(sum - 1.0) < 1e-13);
        }

        // dot(weights, lnDF) == logDF(t) to 1e-13 at all sample points
        for (int t : {30, 45, 60, 90, 120, 150, 180}) {
            auto   w   = li.nodeWeights(t, nodes);
            double dot = 0.0;
            for (std::size_t k = 0; k < n; ++k) dot += w[k] * nodes[k].lnDF;
            double direct = li.logDF(t, nodes);
            assert(std::abs(dot - direct) < 1e-13);
        }
    }

    // ── AveragedQuadraticInterpolator ────────────────────────
    // 6-node test curve gives: first, 3 interior, last intervals.
    {
        // lnDF(T) = -0.05 * T/360 (flat 5% rate, simple compounding approximation)
        std::vector<Node> nodes = {
            { 30,  -0.05 *  30.0/360.0},
            { 90,  -0.05 *  90.0/360.0},
            {180,  -0.05 * 180.0/360.0},
            {360,  -0.05 * 360.0/360.0},
            {720,  -0.05 * 720.0/360.0},
            {1080, -0.05 *1080.0/360.0}
        };
        AveragedQuadraticInterpolator aq;
        std::size_t n = nodes.size(); // 6

        // (1) nodeWeights returns unit vector at every node
        for (std::size_t k = 0; k < n; ++k) {
            auto w = aq.nodeWeights(nodes[k].days, nodes);
            assert(w.size() == n);
            for (std::size_t j = 0; j < n; ++j) {
                if (j == k) assert(std::abs(w[j] - 1.0) < 1e-13);
                else        assert(std::abs(w[j])       < 1e-13);
            }
        }

        // (2) weights sum to 1 everywhere (nodes and non-nodes)
        for (int t : {30, 60, 90, 135, 180, 270, 360, 540, 720, 900, 1080}) {
            auto w = aq.nodeWeights(t, nodes);
            double sum = 0.0;
            for (double x : w) sum += x;
            assert(std::abs(sum - 1.0) < 1e-13);
        }

        // (3) First interval [30,90]: exactly 2 non-zero weights (log-linear fallback)
        for (int t : {45, 60, 75}) {
            auto w = aq.nodeWeights(t, nodes);
            int nz = 0;
            for (double x : w) if (std::abs(x) > 1e-15) ++nz;
            assert(nz == 2);
        }

        // (4) Interior intervals: exactly 4 non-zero weights
        //     Midpoints of [90,180], [180,360], [360,720]  →  i = 1, 2, 3
        for (int t : {135, 270, 540}) {
            auto w = aq.nodeWeights(t, nodes);
            int nz = 0;
            for (double x : w) if (std::abs(x) > 1e-15) ++nz;
            assert(nz == 4);
        }

        // (5) dot(weights, lnDF) == logDF(t) to 1e-13
        for (int t : {30, 45, 90, 135, 180, 270, 360, 540, 720, 900, 1080}) {
            auto   w   = aq.nodeWeights(t, nodes);
            double dot = 0.0;
            for (std::size_t k = 0; k < n; ++k) dot += w[k] * nodes[k].lnDF;
            double direct = aq.logDF(t, nodes);
            assert(std::abs(dot - direct) < 1e-13);
        }
    }

    // ── Swap bootstrap round-trip ────────────────────────────
    // Build a minimal synthetic swap curve and verify that each calibrated
    // node recovers its market par rate to within 1e-12.
    {
        // Three quotes: 90d (≤180, simple), 360d (1Y), 720d (2Y)
        std::vector<MarketQuote> testQ = {
            { 90, 0.04, 0.035},
            {180, 0.04, 0.038},
            {360, 0.04, 0.040},
            {720, 0.04, 0.042},
        };

        for (bool useAQ : {false, true}) {
            auto interp = useAQ
                ? std::unique_ptr<Interpolator>(std::make_unique<AveragedQuadraticInterpolator>())
                : std::unique_ptr<Interpolator>(std::make_unique<LinearInterpolator>());
            YieldCurve sc = buildSwapCurve(testQ, std::move(interp));
            assert(sc.nodes().size() == testQ.size());

            // All DFs should be in (0, 1].
            for (const auto& nd : sc.nodes()) {
                double df = std::exp(nd.lnDF);
                assert(df > 0.0 && df <= 1.0 + 1e-12);
            }

            // Round-trip: for long-maturity nodes, re-compute the par rate
            // from the calibrated curve and compare to the input quote.
            for (std::size_t qi = 0; qi < testQ.size(); ++qi) {
                int T = testQ[qi].days;
                if (T <= 180) continue;
                double dfT = sc.DF(T);
                double sumDFxDCF = 0.0;
                for (int ti = 180; ti <= T; ti += 180)
                    sumDFxDCF += sc.DF(ti) * 0.5;
                double parRecov = (1.0 - dfT) / sumDFxDCF;
                double parInput = testQ[qi].parSwapRate;
                assert(std::abs(parRecov - parInput) < 1e-10);
            }
        }
    }

    std::cout << "All tests passed.\n";
    return 0;
#else
    return curves::run();
#endif
}
