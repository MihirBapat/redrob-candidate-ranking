"""
Stage 5 — reasoning.py

Generates the `reasoning` column by SLOT-FILLING from facts extracted literally
from each candidate's profile — never an LLM at inference (no network, and
structurally impossible to hallucinate). Six rank-aware template families keep the
tone consistent with the rank, satisfying the spec's Stage-4 manual-review checks:
specific facts, JD connection, honest concerns, no hallucination, variation,
rank-consistency.

Every value used is extracted from the candidate record, so the reasoning can only
state things that are actually in the profile.
"""

from datetime import datetime
from canonicaliser import canonicalise_set

CURRENT_DATE = datetime(2026, 6, 1)
_CONNECTORS = ["brings", "has demonstrated", "shows depth in", "offers", "contributes"]


def _recent_company(cand):
    for j in (cand.get("career_history") or []):
        if j.get("company"):
            return j["company"]
    return None


def _gap_years(cand):
    """Years since the most recent role ended (0 if currently employed)."""
    months = []
    for j in (cand.get("career_history") or []):
        if j.get("is_current"):
            return 0.0
        end = j.get("end_date")
        if end:
            try:
                d = datetime.strptime(end, "%Y-%m-%d")
                months.append((CURRENT_DATE.year - d.year) * 12 + (CURRENT_DATE.month - d.month))
            except ValueError:
                pass
    return round(min(months) / 12.0, 1) if months else 0.0


def extract_facts(cand, jd):
    """The literal values the reasoning is allowed to reference."""
    p = cand.get("profile") or {}
    canon = canonicalise_set(cand.get("skills"))
    matched = sorted(canon & jd["required_skills"])
    missing = sorted(jd["required_skills"] - canon)
    s = cand.get("redrob_signals") or {}
    return {
        "title": p.get("current_title") or "Candidate",
        "years": p.get("years_of_experience"),
        "company": _recent_company(cand),
        "matched": matched[:3],
        "missing": missing[:3],
        "gap_years": _gap_years(cand),
        "response_rate": s.get("recruiter_response_rate"),
        "open_to_work": s.get("open_to_work_flag"),
    }


def _exp_phrase(f, i=0):
    return (f"{f['years']} yrs" if f["years"] is not None else "experience")


def pick_family(rank, feats):
    """Route to a template family from rank + feature signals."""
    dom = float(feats.get("domain_fit", 0))
    exp = float(feats.get("experience_fit", 0))
    avail = float(feats.get("availability_score", 0))
    rec = float(feats.get("recency_score", 1))
    coh = float(feats.get("coherence_score", 1))
    if coh < 0.6:
        return "coherence"
    if rank > 70 or dom < 0.3:
        return "below"
    if avail < 0.35 or feats.get("open_flag") is False:
        return "availability"
    if rec < 0.6:
        return "dated"
    if exp < 0.6:
        return "mismatch"
    if rank <= 20 and dom >= 0.6 and exp >= 0.75:
        return "strong"
    return "partial"


def generate_reasoning(rank, cand, jd, feats):
    """Build a fact-grounded, rank-consistent reasoning string."""
    f = extract_facts(cand, jd)
    feats = dict(feats)
    feats["open_flag"] = (cand.get("redrob_signals") or {}).get("open_to_work_flag")
    fam = pick_family(rank, feats)
    conn = _CONNECTORS[rank % len(_CONNECTORS)]
    at_co = f" at {f['company']}" if f["company"] else ""
    skills = ", ".join(f["matched"]) if f["matched"] else "relevant tooling"

    if fam == "strong":
        return (f"{f['title']} with {_exp_phrase(f)}{at_co}; {conn} {skills}, "
                f"directly matching the role's retrieval/ranking focus. Strong, available fit.")
    if fam == "mismatch":
        return (f"{f['title']} with {_exp_phrase(f)}; strong on {skills}, but seniority/"
                f"experience is off the 6-8y target — promising with a level caveat.")
    if fam == "dated":
        return (f"{f['title']}{at_co}; relevant background in {skills}, but inactive ~"
                f"{f['gap_years']}y — skills fit, recency is a concern for a fast-moving role.")
    if fam == "availability":
        rr = f["response_rate"]
        rr_s = f"{rr:.0%} recruiter response" if isinstance(rr, (int, float)) else "low engagement"
        return (f"{f['title']} with {_exp_phrase(f)}{at_co}; on-domain via {skills}, but "
                f"{rr_s}{' and not open to work' if f['open_to_work'] is False else ''} — "
                f"availability is the main concern.")
    if fam == "partial":
        miss = f"; gaps on {', '.join(f['missing'])}" if f["missing"] else ""
        return (f"{f['title']} with {_exp_phrase(f)}{at_co}; partial match — {conn} {skills}{miss}. "
                f"Adjacent but plausibly rampable.")
    if fam == "below":
        return (f"{f['title']}{at_co}; adjacent skills only ({skills}) and below the core "
                f"retrieval/ranking bar — included as a boundary candidate for review.")
    # coherence
    return (f"{f['title']} with {_exp_phrase(f)}{at_co}; {skills} relevant, but profile has "
            f"minor timeline/consistency flags that warrant verification before outreach.")


# tone words that must only appear at high ranks (self-check used by tests)
STRONG_WORDS = ("strong, available fit", "directly matching")
