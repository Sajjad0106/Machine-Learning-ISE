"""
========================================================================================
ML-Based Resume Screening & Intelligent Applicant Tracking System (ATS)
Designed for Information Systems Engineering (ISE) Capstone Evaluation
========================================================================================
Features:
- Dual-mode Operation: 1-Click Instant Demo Benchmarking + Multi-Format File Ingestion
- Advanced NLP Pipeline: TF-IDF Vector Space Model, Sublinear Scaling, n-grams, Cosine Sim
- Comprehensive 350+ Multi-Category Skill Taxonomy with Word-Boundary Regular Expressions
- Multi-Criteria Decision Analysis (MCDA) Scoring: Skills, Semantic Fit, Experience, Education
- Ethical AI Compliance: One-click Blind Screening Mode to mitigate unconscious bias
- Interactive Analytics & Visualizations: Plotly Leaderboard, Radar Charts, Score Gauges
- ATS Health Auditor: Structural completeness, section auditing, and keyword optimization
- Academic Architecture & Viva Preparation Guide for university evaluation
========================================================================================
"""

import io
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from modules.skills_db import ALL_SKILLS, SKILL_TAXONOMY, get_skill_category
from modules.extractor import (
    extract_text_from_file,
    extract_resumes_from_zip,
    audit_resume_structure,
    extract_contact_info,
    extract_candidate_name,
    extract_education_level,
    extract_experience_years,
)
from modules.matcher import score_resumes_batch
from modules.sample_data import SAMPLE_JOB_DESCRIPTIONS, SAMPLE_RESUMES
from modules.visualizations import (
    plot_leaderboard,
    plot_score_breakdown,
    plot_skill_radar,
    plot_score_gauge,
    plot_candidate_comparison,
)
from modules.interview_generator import generate_candidate_interview_guide

# -----------------------------------------------------------------------------
# Streamlit Application Configuration & Custom Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Resume Screening & ATS Intelligence | ISE Project",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, professional enterprise/academic presentation
st.markdown("""
<style>
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        color: #475569;
        letter-spacing: 0.5px;
    }

    /* Skill Badges */
    .badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        margin: 2px 4px 2px 0;
        font-family: Consolas, "Liberation Mono", Menlo, Courier, monospace;
    }
    .badge-match {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .badge-missing {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .badge-bonus {
        background-color: #e0f2fe;
        color: #075985;
        border: 1px solid #7dd3fc;
    }
    .badge-domain {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }

    /* Header banner styling */
    .hero-banner {
        background: #0f172a;
        border-left: 5px solid #2563eb;
        border-radius: 6px;
        padding: 20px 24px;
        color: #f8fafc;
        margin-bottom: 20px;
    }
    .hero-title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
        color: #ffffff;
        letter-spacing: -0.3px;
    }
    .hero-subtitle {
        font-size: 14px;
        color: #94a3b8;
        line-height: 1.5;
    }
    .hero-tags {
        margin-top: 12px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .hero-tag {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        color: #cbd5e1;
        font-family: Consolas, monospace;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "screening_results" not in st.session_state:
    st.session_state.screening_results = None
if "jd_metadata" not in st.session_state:
    st.session_state.jd_metadata = None
if "last_jd_text" not in st.session_state:
    st.session_state.last_jd_text = ""
if "active_candidate_idx" not in st.session_state:
    st.session_state.active_candidate_idx = 0


def generate_html_report(results, jd_meta, threshold):
    rows_html = ""
    for r in results:
        status_cls = "badge-shortlist" if r["composite_score"] >= threshold else "badge-review"
        status_txt = "Shortlisted" if r["composite_score"] >= threshold else "Review Needed"
        matched = ", ".join(r["matched_skills"][:8]) if r["matched_skills"] else "None"
        if len(r["matched_skills"]) > 8:
            matched += f" (+{len(r['matched_skills']) - 8} more)"
        rows_html += f"""
        <tr>
            <td><strong>#{r['rank']}</strong></td>
            <td><strong>{r['candidate_name']}</strong></td>
            <td>{r['composite_score']}%</td>
            <td><span class="{status_cls}">{status_txt}</span></td>
            <td>{r['scores_breakdown']['skill_score']}%</td>
            <td>{r['scores_breakdown']['semantic_score']}%</td>
            <td>{r['experience_years']} yrs</td>
            <td>{r['education']['highest_level']}</td>
            <td>{matched}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Resume Screening Evaluation Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 30px; color: #1e293b; background: #fff; }}
    h1 {{ color: #0f172a; margin-bottom: 4px; font-size: 22px; }}
    .meta {{ color: #64748b; font-size: 13px; margin-bottom: 20px; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
    th, td {{ border: 1px solid #cbd5e1; padding: 8px 10px; text-align: left; font-size: 12px; }}
    th {{ background-color: #f1f5f9; font-weight: 600; color: #334155; }}
    tr:nth-child(even) {{ background-color: #f8fafc; }}
    .badge-shortlist {{ background-color: #dcfce7; color: #166534; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
    .badge-review {{ background-color: #fee2e2; color: #991b1b; padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
    .footer {{ margin-top: 30px; font-size: 11px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 10px; }}
    @media print {{
        body {{ margin: 15px; }}
    }}
</style>
</head>
<body>
    <h1>Candidate Screening & Evaluation Report</h1>
    <div class="meta">
        <strong>Information Systems Engineering (ISE) Decision Support</strong> | 
        Shortlist Threshold: {threshold}% | 
        Total Evaluated: {len(results)} | 
        Shortlisted: {sum(1 for r in results if r['composite_score'] >= threshold)}
    </div>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Candidate</th>
                <th>Score</th>
                <th>Status</th>
                <th>Skill %</th>
                <th>Semantic %</th>
                <th>Exp</th>
                <th>Education</th>
                <th>Top Matched Skills</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    <div class="footer">
        Generated by ML Resume Screening ATS Decision Support System. Use Ctrl+P to print or save as PDF.
    </div>
</body>
</html>"""
    return html


# -----------------------------------------------------------------------------
# Top Hero Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">ML-Based Resume Screening & Applicant Tracking System</div>
    <div class="hero-subtitle">
        An Explainable AI (XAI) decision-support system utilizing TF-IDF Vector Space Modeling, 
        Cosine Similarity, Categorized Skill Taxonomy NER, and Multi-Criteria Decision Analysis (MCDA).
    </div>
    <div class="hero-tags">
        <span class="hero-tag">[SYSTEM] ISE Final Capstone</span>
        <span class="hero-tag">[MODEL] TF-IDF Vector Space</span>
        <span class="hero-tag">[ETHICS] Blind Screening Compliance</span>
        <span class="hero-tag">[DECISION] Multi-Criteria MCDA</span>
        <span class="hero-tag">[AUDIT] Real-Time ATS Analyzer</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Sidebar: Scoring Weights & Model Settings
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Screening Configuration")
    
    st.markdown("**Decision Threshold**")
    threshold = st.slider(
        "Shortlist Review Cutoff (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="Candidates with a composite score equal to or above this threshold are marked for priority interview shortlisting."
    )

    with st.expander("Multi-Factor Decision Weights (MCDA)", expanded=False):
        st.caption("Adjust the contribution of each dimension to the composite score:")
        w_skills = st.slider("Technical Skills Match (%)", 0, 100, 45, 5)
        w_semantic = st.slider("Semantic Similarity (%)", 0, 100, 35, 5)
        w_exp = st.slider("Experience Alignment (%)", 0, 100, 10, 5)
        w_edu = st.slider("Education Level (%)", 0, 100, 10, 5)

        total_weight = w_skills + w_semantic + w_exp + w_edu
        if total_weight > 0:
            weights_dict = {
                "skills": w_skills / total_weight,
                "semantic": w_semantic / total_weight,
                "experience": w_exp / total_weight,
                "education": w_edu / total_weight,
            }
            st.caption(f"Weights normalized to 100% (Sum: {total_weight}%)")
        else:
            weights_dict = {"skills": 0.45, "semantic": 0.35, "experience": 0.10, "education": 0.10}

    st.markdown("---")
    st.markdown("**Ethical AI & Blind Screening**")
    blind_screening = st.toggle(
        "Enable Blind Screening Mode",
        value=False,
        help="Anonymizes candidate names, emails, phone numbers, and identifying links to prevent unconscious demographic bias in line with modern HR compliance (NYC Local Law 144 / EU AI Act)."
    )

    st.markdown("---")
    st.markdown("**Custom Skill Taxonomy**")
    custom_skills_input = st.text_area(
        "Add Custom Skills (comma-separated)",
        placeholder="e.g. LangGraph, Ray, vLLM, Bun, Supabase",
        help="Augment the built-in 350+ skill taxonomy with domain-specific or emerging keywords."
    )

    st.markdown("---")
    st.markdown("### System Specifications")
    st.caption(f"- Skill Database: {len(ALL_SKILLS)} Built-in terms")
    st.caption(f"- Supported Formats: PDF, DOCX, TXT")
    st.caption(f"- Scoring Engine: TF-IDF (1, 2 n-grams) + Cosine Sim")


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_screen, tab_candidate, tab_cohort, tab_ats, tab_docs = st.tabs([
    "Batch Screening & Ranking",
    "Candidate Deep-Dive",
    "Cohort Analytics",
    "Single Resume ATS Optimizer",
    "System Architecture & ISE Guide"
])


# =============================================================================
# TAB 1: BATCH SCREENING PIPELINE
# =============================================================================
with tab_screen:
    st.subheader("1. Job Description & Candidate Resumes")

    # Workflow mode selector
    mode = st.radio(
        "Choose Evaluation Mode:",
        ["Instant Demo Benchmarking (ISE Showcase)", "Custom Job Description & Resume Uploads"],
        horizontal=True
    )

    job_description = ""
    resume_corpus = {}

    if "Instant Demo" in mode:
        st.info("Showcase Mode: Select a pre-configured role template and run our benchmark batch of 5 candidate resumes (representing diverse seniority and skill alignments) with one click.")
        
        col_jd_pick, col_action = st.columns([2, 1])
        with col_jd_pick:
            selected_role = st.selectbox(
                "Select Pre-configured Job Role Template:",
                list(SAMPLE_JOB_DESCRIPTIONS.keys())
            )
            job_description = SAMPLE_JOB_DESCRIPTIONS[selected_role].strip()

        with col_action:
            st.markdown("<br>", unsafe_allow_html=True)
            run_demo = st.button("Run Live Screening Benchmark", type="primary", use_container_width=True)

        with st.expander("View / Edit Active Job Description", expanded=False):
            job_description = st.text_area("Job Description Text", value=job_description, height=180)

        if run_demo:
            resume_corpus = SAMPLE_RESUMES.copy()

    else:
        st.markdown("Enter custom job requirements and upload your applicant resumes (PDF, DOCX, or TXT):")
        
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            template_helper = st.selectbox(
                "Optional: Load a template into the Job Description editor:",
                ["-- Select a Template (Optional) --"] + list(SAMPLE_JOB_DESCRIPTIONS.keys())
            )
            default_jd = SAMPLE_JOB_DESCRIPTIONS[template_helper] if template_helper in SAMPLE_JOB_DESCRIPTIONS else ""
            job_description = st.text_area(
                "Paste Job Description:",
                value=default_jd,
                height=220,
                placeholder="Paste the job responsibilities, minimum qualifications, required technical skills, and years of experience..."
            )

        with col_t2:
            st.markdown("<br>", unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "Upload Candidate Resumes (PDF, DOCX, TXT, or ZIP archive):",
                type=["pdf", "docx", "txt", "zip"],
                accept_multiple_files=True
            )
            
            run_custom = st.button("Analyze & Rank Uploaded Resumes", type="primary", use_container_width=True)

        if run_custom:
            if not job_description.strip():
                st.error("Error: Please provide a Job Description.")
            elif not uploaded_files:
                st.error("Error: Please upload at least one candidate resume (PDF, DOCX, TXT, or ZIP).")
            else:
                with st.spinner("Extracting text and running parsing..."):
                    extraction_errors = []
                    for uf in uploaded_files:
                        try:
                            if uf.name.lower().endswith(".zip"):
                                zip_resumes = extract_resumes_from_zip(uf.getvalue())
                                if zip_resumes:
                                    resume_corpus.update(zip_resumes)
                                else:
                                    extraction_errors.append(f"{uf.name}: No valid PDF, DOCX, or TXT files found in ZIP.")
                            else:
                                txt = extract_text_from_file(uf)
                                if txt.strip():
                                    resume_corpus[uf.name] = txt
                                else:
                                    extraction_errors.append(f"{uf.name}: File contained no extractable text.")
                        except Exception as e:
                            extraction_errors.append(f"{uf.name}: {str(e)}")

                    if extraction_errors:
                        with st.expander(f"Extraction Warnings ({len(extraction_errors)})", expanded=True):
                            for err in extraction_errors:
                                st.warning(err)

    # -------------------------------------------------------------------------
    # Execution & Results Calculation
    # -------------------------------------------------------------------------
    if resume_corpus and job_description.strip():
        with st.spinner("Running NLP pipeline, TF-IDF vectorization, skill matching, and MCDA scoring..."):
            custom_skills = [s.strip() for s in custom_skills_input.split(",") if s.strip()]
            results, jd_meta = score_resumes_batch(
                job_description=job_description,
                resumes=resume_corpus,
                custom_skills=custom_skills,
                weights=weights_dict,
                anonymize=blind_screening
            )
            st.session_state.screening_results = results
            st.session_state.jd_metadata = jd_meta
            st.session_state.last_jd_text = job_description
            st.success(f"Screening complete: Successfully ranked {len(results)} candidate resumes.")

    # -------------------------------------------------------------------------
    # Render Screening Dashboard if results exist
    # -------------------------------------------------------------------------
    if st.session_state.screening_results is not None:
        results = st.session_state.screening_results
        jd_meta = st.session_state.jd_metadata
        
        st.markdown("---")
        st.subheader("2. Candidate Screening & Ranking Leaderboard")

        # Top Executive Metrics Row
        shortlisted_count = sum(1 for r in results if r["composite_score"] >= threshold)
        top_cand = results[0]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Resumes Screened", len(results))
        m2.metric("Shortlisted for Review", f"{shortlisted_count} / {len(results)}", f"{round(shortlisted_count/len(results)*100)}%")
        m3.metric("Required Skills in JD", len(jd_meta["required_skills"]))
        m4.metric("Top Candidate Match", f"{top_cand['composite_score']}%", top_cand['candidate_name'])

        # Required Skills Chips
        with st.expander(f"Skills Detected in Job Description ({len(jd_meta['required_skills'])})", expanded=False):
            if jd_meta["required_skills"]:
                chips_html = "".join([f'<span class="badge badge-domain">{s}</span>' for s in jd_meta["required_skills"]])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.info("No recognized technical skills were extracted from the JD. Ranking relied on TF-IDF semantic vector space modeling.")

        # Interactive Leaderboard Plotly Chart
        chart_col, filter_col = st.columns([3, 1])
        with filter_col:
            st.markdown("#### Filter & View Options")
            tier_filter = st.selectbox(
                "Filter by Review Status:",
                ["All Candidates", f"Above Threshold ({threshold}%+)", "Tier 1 (Strong)", "Needs Review (<50%)"]
            )
            search_kw = st.text_input("Search Candidate / Skill:", placeholder="e.g. Python, Docker, Alex")
            st.caption(f"**Applied Weights:**<br>Skills: {jd_meta['weights_used']['skills']}%<br>Semantic: {jd_meta['weights_used']['semantic']}%<br>Experience: {jd_meta['weights_used']['experience']}%<br>Education: {jd_meta['weights_used']['education']}%", unsafe_allow_html=True)

        with chart_col:
            st.plotly_chart(plot_leaderboard(results), use_container_width=True)

        # Tabular View
        filtered_results = results
        if tier_filter == f"Above Threshold ({threshold}%+)":
            filtered_results = [r for r in results if r["composite_score"] >= threshold]
        elif tier_filter == "Tier 1 (Strong)":
            filtered_results = [r for r in results if r["composite_score"] >= 80]
        elif tier_filter == "Needs Review (<50%)":
            filtered_results = [r for r in results if r["composite_score"] < 50]

        if search_kw.strip():
            kw = search_kw.strip().lower()
            filtered_results = [
                r for r in filtered_results
                if kw in r["candidate_name"].lower()
                or any(kw in s.lower() for s in r["candidate_skills"])
                or kw in r["education"]["highest_level"].lower()
            ]

        table_rows = []
        for r in filtered_results:
            table_rows.append({
                "Rank": f"#{r['rank']}",
                "Candidate": r["candidate_name"],
                "Composite Score": f"{r['composite_score']}%",
                "Review Status": "Shortlisted" if r["composite_score"] >= threshold else "Review Needed",
                "Tier": r["tier"],
                "Skill Match": f"{r['scores_breakdown']['skill_score']}%",
                "Semantic Fit": f"{r['scores_breakdown']['semantic_score']}%",
                "Raw Cosine": f"{r['scores_breakdown']['raw_cosine']}%",
                "Experience": f"{r['experience_years']} yrs",
                "Highest Education": r["education"]["highest_level"],
                "Matched Skills Count": len(r["matched_skills"]),
                "Missing Skills Count": len(r["missing_skills"])
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Export Actions
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            csv_data = pd.DataFrame(table_rows).to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "Download Results (CSV)",
                data=csv_data,
                file_name="resume_screening_leaderboard.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_exp2:
            summary_text = f"RESUME SCREENING SHORTLIST REPORT\nGenerated: 2026-11-08\nTotal Evaluated: {len(results)}\nReview Threshold: {threshold}%\n\n"
            for r in results:
                status = "SHORTLISTED" if r["composite_score"] >= threshold else "NEEDS REVIEW"
                summary_text += f"Rank #{r['rank']} | {r['candidate_name']} | Score: {r['composite_score']}% | {status}\n"
                summary_text += f"  Matched: {', '.join(r['matched_skills']) if r['matched_skills'] else 'None'}\n"
                summary_text += f"  Missing: {', '.join(r['missing_skills']) if r['missing_skills'] else 'None'}\n\n"
            
            st.download_button(
                "Download Shortlist (.TXT)",
                data=summary_text.encode('utf-8'),
                file_name="executive_shortlist_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col_exp3:
            html_report = generate_html_report(results, jd_meta, threshold)
            st.download_button(
                "Download Printable Report (HTML)",
                data=html_report.encode('utf-8'),
                file_name="candidate_screening_report.html",
                mime="text/html",
                use_container_width=True
            )


# =============================================================================
# TAB 2: CANDIDATE DEEP-DIVE & GAP ANALYSIS
# =============================================================================
with tab_candidate:
    if st.session_state.screening_results is None:
        st.info("Please run a screening analysis in Tab 1 (or run the Instant Demo) to inspect candidate profiles.")
    else:
        results = st.session_state.screening_results
        
        # Candidate Selector
        candidate_options = [f"#{r['rank']} - {r['candidate_name']} ({r['composite_score']}%)" for r in results]
        selected_cand_str = st.selectbox("Select Candidate to Inspect:", candidate_options)
        selected_idx = candidate_options.index(selected_cand_str)
        cand = results[selected_idx]

        st.markdown("---")

        # Profile Overview Card
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown(f"### Candidate Profile: {cand['candidate_name']}")
            st.markdown(f"**Application File:** `{cand['filename']}` | **Rank:** `#{cand['rank']} of {len(results)}`")
            
            # Contact Information Badges
            contact = cand["contact"]
            contact_items = []
            if contact.get("email"):
                contact_items.append(f"**Email:** {contact['email']}")
            if contact.get("phone"):
                contact_items.append(f"**Phone:** {contact['phone']}")
            if contact.get("linkedin"):
                contact_items.append(f"**LinkedIn:** {contact['linkedin']}")
            if contact.get("github"):
                contact_items.append(f"**GitHub:** {contact['github']}")
            if contact.get("portfolio"):
                contact_items.append(f"**Portfolio:** {contact['portfolio']}")

            if contact_items:
                st.markdown(" | ".join(contact_items))
            else:
                st.caption("No direct contact links extracted.")

            st.markdown(f"**Experience:** `{cand['experience_years']} Years` | **Highest Degree:** `{cand['education']['highest_level']}` ({', '.join(cand['education']['majors']) if cand['education']['majors'] else 'General'})")

        with c_right:
            st.plotly_chart(plot_score_gauge(cand["composite_score"], "Relevance Match"), use_container_width=True)

        # Dimension Breakdown Tiles
        b1, b2, b3, b4 = st.columns(4)
        b1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Skill Coverage</div>
            <div class="metric-value">{cand['scores_breakdown']['skill_score']}%</div>
            <div style="font-size: 11px; color: #64748b;">{len(cand['matched_skills'])} of {len(st.session_state.jd_metadata['required_skills'])} required</div>
        </div>
        """, unsafe_allow_html=True)

        b2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Semantic Alignment</div>
            <div class="metric-value">{cand['scores_breakdown']['semantic_score']}%</div>
            <div style="font-size: 11px; color: #64748b;">Raw Cosine: {cand['scores_breakdown']['raw_cosine']}%</div>
        </div>
        """, unsafe_allow_html=True)

        b3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Experience Match</div>
            <div class="metric-value">{cand['scores_breakdown']['exp_score']}%</div>
            <div style="font-size: 11px; color: #64748b;">Detected: {cand['experience_years']} Years</div>
        </div>
        """, unsafe_allow_html=True)

        b4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Education Rank</div>
            <div class="metric-value">{cand['scores_breakdown']['edu_score']}%</div>
            <div style="font-size: 11px; color: #64748b;">{cand['education']['highest_level']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Detailed Skill Gap Analysis
        st.subheader("Skill Gap & Domain Competency Matrix")
        sk_col1, sk_col2, sk_col3 = st.columns(3)

        with sk_col1:
            st.markdown(f"**[+] Matched Required Skills ({len(cand['matched_skills'])})**")
            if cand["matched_skills"]:
                chips = "".join([f'<span class="badge badge-match">{s}</span>' for s in cand["matched_skills"]])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.write("None detected.")

        with sk_col2:
            st.markdown(f"**[-] Missing Skills from JD ({len(cand['missing_skills'])})**")
            if cand["missing_skills"]:
                chips = "".join([f'<span class="badge badge-missing">{s}</span>' for s in cand["missing_skills"]])
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.write("Full coverage: No missing required skills.")

        with sk_col3:
            st.markdown(f"**[*] Additional / Complementary Skills ({len(cand['additional_skills'])})**")
            if cand["additional_skills"]:
                chips = "".join([f'<span class="badge badge-bonus">{s}</span>' for s in cand["additional_skills"][:15]])
                if len(cand["additional_skills"]) > 15:
                    chips += f'<span class="badge badge-bonus">+{len(cand["additional_skills"]) - 15} more</span>'
                st.markdown(chips, unsafe_allow_html=True)
            else:
                st.write("No additional domain skills detected.")

        # Radar chart & ATS Structural Audit
        st.markdown("<br>", unsafe_allow_html=True)
        r_col, audit_col = st.columns([1, 1])

        with r_col:
            st.markdown("#### Domain Competency Radar Analysis")
            st.plotly_chart(plot_skill_radar(cand["category_breakdown"]), use_container_width=True)

        with audit_col:
            st.markdown("#### ATS Structural & Formatting Audit")
            audit = cand["resume_audit"]
            st.metric("ATS Health Score", f"{audit['ats_health_score']} / 100")
            st.write(f"- Word Count: {audit['word_count']} words")
            st.write(f"- Bullet Points Detected: {audit['bullet_count']}")
            st.write(f"- Sections Identified: {', '.join(audit['found_sections'])}")
            if audit['missing_sections']:
                st.write(f"- Sections Missing: {', '.join(audit['missing_sections'])}")

            if audit["suggestions"]:
                st.markdown("**Actionable Suggestions for Candidate:**")
                for sug in audit["suggestions"]:
                    st.info(f"Recommendation: {sug}")

        # Parsed Text Drawer
        with st.expander("Extracted Resume Text (Raw)", expanded=False):
            st.text_area("Parsed Text Content", value=cand["_raw_text"], height=300, disabled=True)

        # Head-to-Head Comparison
        st.markdown("---")
        with st.expander("Head-to-Head Candidate Comparison (Compare Two Finalists)", expanded=False):
            st.markdown("#### Direct Head-to-Head Comparison")
            cmp_col1, cmp_col2 = st.columns(2)
            with cmp_col1:
                cand_a_idx = st.selectbox(
                    "Select First Candidate:",
                    range(len(results)),
                    format_func=lambda i: f"#{results[i]['rank']} - {results[i]['candidate_name']} ({results[i]['composite_score']}%)",
                    index=0,
                    key="cmp_cand_a"
                )
            with cmp_col2:
                cand_b_idx = st.selectbox(
                    "Select Second Candidate:",
                    range(len(results)),
                    format_func=lambda i: f"#{results[i]['rank']} - {results[i]['candidate_name']} ({results[i]['composite_score']}%)",
                    index=min(1, len(results) - 1),
                    key="cmp_cand_b"
                )

            c_a = results[cand_a_idx]
            c_b = results[cand_b_idx]

            st.plotly_chart(plot_candidate_comparison(c_a, c_b), use_container_width=True)

            # Skill overlap & differential breakdown
            skills_a = set(c_a["candidate_skills"])
            skills_b = set(c_b["candidate_skills"])

            diff_col1, diff_col2, diff_col3 = st.columns(3)
            with diff_col1:
                st.markdown(f"**Unique to {c_a['candidate_name']} ({len(skills_a - skills_b)})**")
                unique_a = sorted(list(skills_a - skills_b))
                if unique_a:
                    st.markdown("".join([f'<span class="badge badge-match">{s}</span>' for s in unique_a[:12]]), unsafe_allow_html=True)
                else:
                    st.write("None")
            with diff_col2:
                st.markdown(f"**Shared Skills ({len(skills_a & skills_b)})**")
                shared = sorted(list(skills_a & skills_b))
                if shared:
                    st.markdown("".join([f'<span class="badge badge-domain">{s}</span>' for s in shared[:12]]), unsafe_allow_html=True)
                else:
                    st.write("None")
            with diff_col3:
                st.markdown(f"**Unique to {c_b['candidate_name']} ({len(skills_b - skills_a)})**")
                unique_b = sorted(list(skills_b - skills_a))
                if unique_b:
                    st.markdown("".join([f'<span class="badge badge-bonus">{s}</span>' for s in unique_b[:12]]), unsafe_allow_html=True)
                else:
                    st.write("None")

        # Personalized Technical Interview Guide
        st.markdown("---")
        with st.expander("Personalized Technical Interview Guide (Auto-Generated for this Candidate)", expanded=False):
            guide = generate_candidate_interview_guide(cand, st.session_state.jd_metadata)
            st.markdown(f"#### Interview Preparation Guide: {cand['candidate_name']} (Composite Match: {cand['composite_score']}%)")
            st.caption("Auto-generated questions targeted at candidate's verified skills, identified skill gaps, and behavioral competencies.")

            if guide["matched_questions"]:
                st.markdown("**Questions to Verify Claimed Strengths:**")
                for item in guide["matched_questions"]:
                    st.markdown(f"- **[{item['domain']}]** {item['question']}")
                    st.caption(f"  *Evaluation Guide:* {item['evaluation_guide']}")

            if guide["missing_questions"]:
                st.markdown("**Questions to Probe Skill Gaps:**")
                for item in guide["missing_questions"]:
                    st.markdown(f"- **[{item['domain']}]** {item['question']}")
                    st.caption(f"  *Evaluation Guide:* {item['evaluation_guide']}")

            st.markdown("**Core Behavioral & Situational Questions:**")
            for q, g in guide["behavioral_questions"]:
                st.markdown(f"- {q}")
                st.caption(f"  *Evaluation Guide:* {g}")

            # Download Interview Guide TXT
            guide_txt = f"TECHNICAL INTERVIEW GUIDE: {cand['candidate_name']}\nComposite Score: {cand['composite_score']}%\n\n"
            guide_txt += "1. VERIFY CLAIMED STRENGTHS:\n"
            for item in guide["matched_questions"]:
                guide_txt += f"Q: {item['question']}\nGuide: {item['evaluation_guide']}\n\n"
            guide_txt += "2. PROBE SKILL GAPS:\n"
            for item in guide["missing_questions"]:
                guide_txt += f"Q: {item['question']}\nGuide: {item['evaluation_guide']}\n\n"
            guide_txt += "3. BEHAVIORAL QUESTIONS:\n"
            for q, g in guide["behavioral_questions"]:
                guide_txt += f"Q: {q}\nGuide: {g}\n\n"

            st.download_button(
                f"Download Interview Guide for {cand['candidate_name']} (.TXT)",
                data=guide_txt.encode("utf-8"),
                file_name=f"interview_guide_{cand['candidate_name'].replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )


# =============================================================================
# TAB 3: COHORT ANALYTICS & INSIGHTS
# =============================================================================
with tab_cohort:
    if st.session_state.screening_results is None:
        st.info("Please run a screening analysis in Tab 1 to generate cohort intelligence.")
    else:
        results = st.session_state.screening_results
        jd_meta = st.session_state.jd_metadata

        st.subheader("Applicant Cohort Intelligence")

        # Multi-Factor Dimension Breakdown
        st.plotly_chart(plot_score_breakdown(results), use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Candidate Tier Distribution")
            tiers = [r["tier"] for r in results]
            tier_df = pd.Series(tiers).value_counts().reset_index()
            tier_df.columns = ["Tier", "Candidate Count"]
            st.dataframe(tier_df, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("#### High-Frequency Skills in Applicant Pool")
            all_pool_skills = []
            for r in results:
                all_pool_skills.extend(r["candidate_skills"])
            if all_pool_skills:
                skill_freq = pd.Series(all_pool_skills).value_counts().head(10).reset_index()
                skill_freq.columns = ["Skill", "Found in Resumes"]
                st.dataframe(skill_freq, use_container_width=True, hide_index=True)
            else:
                st.write("No skills data available.")


# =============================================================================
# TAB 4: SINGLE RESUME ATS OPTIMIZER
# =============================================================================
with tab_ats:
    st.subheader("Real-Time ATS Resume Auditor & Optimizer")
    st.markdown("Evaluate an individual resume against applicant tracking system standards to discover missing sections, formatting weaknesses, and keyword density.")

    ats_input_mode = st.radio("Input method:", ["Upload File", "Paste Text"], horizontal=True)
    resume_audit_text = ""

    if ats_input_mode == "Upload File":
        single_file = st.file_uploader("Upload Resume for ATS Check (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"], key="single_ats_file")
        if single_file:
            try:
                resume_audit_text = extract_text_from_file(single_file)
            except Exception as e:
                st.error(f"Error parsing file: {e}")
    else:
        resume_audit_text = st.text_area("Paste Resume Text:", height=250, placeholder="Paste your resume contents here...")

    if st.button("Audit Resume for ATS Readability", type="primary"):
        if not resume_audit_text.strip():
            st.error("Please provide resume text to audit.")
        else:
            with st.spinner("Analyzing document structure, formatting, and keyword density..."):
                audit_result = audit_resume_structure(resume_audit_text)
                contact_res = extract_contact_info(resume_audit_text)
                skills_found = [s for s in ALL_SKILLS if f" {s} " in f" {resume_audit_text.lower()} "]
                
                col_g, col_details = st.columns([1, 2])
                with col_g:
                    st.plotly_chart(plot_score_gauge(audit_result["ats_health_score"], "ATS Score"), use_container_width=True)

                with col_details:
                    st.markdown("### Document Structure Summary")
                    st.write(f"- Word Count: {audit_result['word_count']} words " + 
                             ("[Optimal length]" if 400 <= audit_result['word_count'] <= 1200 else "[Consider adjusting length]"))
                    st.write(f"- Bullet Points Found: {audit_result['bullet_count']}")
                    st.write(f"- Recognized Skills Identified: {len(skills_found)}")
                    st.write(f"- Email Extracted: {contact_res['email'] or 'Not Found'}")
                    st.write(f"- Phone Extracted: {contact_res['phone'] or 'Not Found'}")

                st.markdown("---")
                st.markdown("#### Standard ATS Sections Checklist")
                for sec in audit_result["found_sections"]:
                    st.success(f"[Present] {sec}")
                for sec in audit_result["missing_sections"]:
                    st.error(f"[Missing] {sec}")

                if audit_result["suggestions"]:
                    st.markdown("#### Actionable Improvement Recommendations")
                    for s in audit_result["suggestions"]:
                        st.warning(f"Recommendation: {s}")


# =============================================================================
# TAB 5: SYSTEM ARCHITECTURE & ISE PROJECT GUIDE
# =============================================================================
with tab_docs:
    st.subheader("ISE Software Engineering & ML Architecture Documentation")
    st.caption("Comprehensive academic documentation, algorithmic foundations, and Viva Voce preparation guide.")

    with st.expander("1. Software Engineering Architecture & Pipeline Design", expanded=True):
        st.markdown(r"""
        ### System Architecture & Data Flow
        The system follows a modular, 5-stage Machine Learning & Information Retrieval pipeline:
        
        1. **Document Ingestion Layer:**
           - Multi-format ingestion supporting `.pdf` (via stream binary parsing with `pypdf`), `.docx` (via XML paragraph and table extraction with `python-docx`), and `.txt` (multi-encoding fallback).
        2. **Preprocessing & Normalization Layer:**
           - Character cleansing, unicode normalization, tokenization, and preservation of programming language tokens (e.g. `C++`, `C#`, `.NET`, `Node.js`, `CI/CD`).
        3. **NLP Vector Space Modeling:**
           - Unigram and Bigram `(1, 2)` TF-IDF Vectorizer with Sublinear Term Frequency ($w_{t,d} = 1 + \ln(tf)$).
           - High-dimensional Cosine Similarity computation in vector space.
        4. **Multi-Category Skill Extraction (NER / Taxonomy):**
           - Exact boundary-safe regex matching against a 350+ industry-standard skill taxonomy categorized into 9 domains.
        5. **Multi-Criteria Decision Analysis (MCDA):**
           - Weighted composite scoring model blending:
             $$\text{Composite Score} = w_{\text{skill}} \cdot S_{\text{skill}} + w_{\text{semantic}} \cdot S_{\text{semantic}} + w_{\text{exp}} \cdot S_{\text{exp}} + w_{\text{edu}} \cdot S_{\text{edu}}$$
        """)

    with st.expander("2. Mathematical & Algorithmic Formulation", expanded=True):
        st.markdown(r"""
        ### 1. TF-IDF Formulation (Term Frequency - Inverse Document Frequency)
        Term Frequency with sublinear scaling prevents long resumes with high keyword repetition from artificially dominating the score:
        $$\text{TF}(t, d) = \begin{cases} 1 + \ln(\text{count}(t, d)) & \text{if } \text{count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

        Inverse Document Frequency downweights universally common words:
        $$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

        ### 2. High-Dimensional Cosine Similarity
        Measures the angle between the Job Description vector $\vec{J}$ and Resume vector $\vec{R}$ independently of document length:
        $$\text{Cosine Similarity}(\vec{J}, \vec{R}) = \frac{\vec{J} \cdot \vec{R}}{\|\vec{J}\| \|\vec{R}\|} = \frac{\sum_{i=1}^{n} J_i R_i}{\sqrt{\sum_{i=1}^{n} J_i^2} \sqrt{\sum_{i=1}^{n} R_i^2}}$$

        ### 3. Skill Coverage Metric
        Let $S_{\text{req}}$ be the set of skills extracted from the Job Description, and $S_{\text{cand}}$ be the set of skills extracted from the candidate's resume:
        $$\text{Skill Coverage} = \frac{|S_{\text{req}} \cap S_{\text{cand}}|}{|S_{\text{req}}|}$$
        """)

    with st.expander("3. Ethical AI, Fairness & Regulatory Compliance", expanded=True):
        st.markdown("""
        ### Why Transparent NLP Beats Supervised Black-Box AI in Hiring:
        - **The Amazon Hiring Tool Precedent:** In 2018, Amazon notoriously decommissioned an internal hiring AI model because it was trained on 10 years of historical tech hiring data (predominantly male), causing the neural network to penalize resumes containing the word *"women's"* (e.g. *"women's chess club captain"*).
        - **Explainable AI (XAI):** This application utilizes transparent, verifiable feature extraction where every score component (skill coverage, semantic overlap, years of experience) is directly interpretable by human HR auditors.
        - **Blind Screening Mode:** Toggling Blind Screening strips candidate names, emails, phone numbers, and demographics to ensure initial ranking is purely based on competence and qualifications.
        - **Regulatory Alignment:** Adheres to **NYC Local Law 144** (Automated Employment Decision Tools audit requirement) and the **EU AI Act (High-Risk AI Systems)** guidelines for recruitment.
        """)

    with st.expander("4. ISE Viva Voce Examination Cheat Sheet (Top Questions & Answers)", expanded=True):
        st.markdown(r"""
        **Q1: Why did you choose TF-IDF + Cosine Similarity instead of training a deep classification model?**
        > *Answer:* Unsupervised TF-IDF + Cosine Similarity does not require a large, labeled historical hiring dataset. Historical hiring datasets are notoriously difficult to acquire lawfully, and frequently embed historical racial, gender, or institutional hiring biases. TF-IDF provides an explainable, deterministic baseline that evaluates text alignment transparently without requiring subjective labels.

        **Q2: What is sublinear TF scaling and why is it important here?**
        > *Answer:* Sublinear TF scaling replaces raw term frequency with $1 + \ln(\text{count})$. This prevents "keyword stuffing" where an applicant writes the word "Python" 50 times in white text to manipulate the ATS parser. Twenty mentions of a skill does not mean an applicant is twenty times more proficient.

        **Q3: How does your system handle different file formats?**
        > *Answer:* We developed a modular extraction layer that identifies the file extension: PDF files are parsed page-by-page via `pypdf`, DOCX files parse paragraph and table structures via `python-docx`, and TXT files are processed with multi-encoding fallback (`utf-8`, `utf-16`, `latin-1`).

        **Q4: How do you address algorithmic bias?**
        > *Answer:* We implemented an Ethical AI Blind Screening mode that removes PII (Personally Identifiable Information: names, emails, phones, social links) during candidate ranking. Furthermore, we provide a full score breakdown so humans can audit why any candidate was ranked at a specific tier.
        """)
