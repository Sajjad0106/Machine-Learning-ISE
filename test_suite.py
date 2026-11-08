import io
import zipfile
import pandas as pd
from modules.skills_db import ALL_SKILLS, get_skill_category
from modules.extractor import (
    extract_text_from_file,
    extract_resumes_from_zip,
    audit_resume_structure,
    extract_contact_info,
    extract_education_level,
    extract_experience_years
)
from modules.matcher import score_resumes_batch
from modules.sample_data import SAMPLE_JOB_DESCRIPTIONS, SAMPLE_RESUMES
from modules.visualizations import (
    plot_leaderboard,
    plot_score_breakdown,
    plot_skill_radar,
    plot_score_gauge,
    plot_candidate_comparison
)
from modules.interview_generator import generate_candidate_interview_guide
from app import generate_html_report

print("1. Testing all 5 sample Job Descriptions against sample resumes...")
for role, jd in SAMPLE_JOB_DESCRIPTIONS.items():
    res, meta = score_resumes_batch(jd, SAMPLE_RESUMES)
    assert len(res) == len(SAMPLE_RESUMES), "Count mismatch"
    top = res[0]
    print(f"   [PASS] {role}: Top candidate = {top['candidate_name']} ({top['composite_score']}%)")

print("\n2. Testing Blind Screening mode...")
res_blind, _ = score_resumes_batch(SAMPLE_JOB_DESCRIPTIONS['Machine Learning & AI Engineer'], SAMPLE_RESUMES, anonymize=True)
assert res_blind[0]['candidate_name'].startswith("Candidate #"), "Blind screening name masking failed"
assert res_blind[0]['contact']['email'] == "[Protected for Blind Review]", "Email not protected"
print("   [PASS] Blind screening works: names masked and PII protected!")

print("\n3. Testing DOCX extraction...")
docx_txt = extract_text_from_file("sample_resumes/Alex_Chen_Senior_ML_Engineer.docx")
assert len(docx_txt) > 500, "DOCX parsing failed"
print(f"   [PASS] DOCX text parsed successfully ({len(docx_txt)} characters)")

print("\n4. Testing ATS Resume Audit...")
audit = audit_resume_structure(docx_txt)
assert audit["ats_health_score"] > 0, "ATS audit failed"
print(f"   [PASS] ATS health score: {audit['ats_health_score']}/100, Word count: {audit['word_count']}")

print("\n5. Testing ZIP archive extraction...")
z_buf = io.BytesIO()
with zipfile.ZipFile(z_buf, "w") as z:
    z.writestr("applicant_1.txt", "Senior Python Engineer with 6 years experience in Django and AWS.")
    z.writestr("applicant_2.txt", "Data Scientist with PyTorch, NLP, and Machine Learning background.")
z_buf.seek(0)
unpacked = extract_resumes_from_zip(z_buf.getvalue())
assert len(unpacked) == 2, "ZIP extraction failed"
print("   [PASS] ZIP archive unpacker extracted all resumes successfully!")

print("\n6. Testing Head-to-Head Candidate Comparison visualization...")
res_web, meta_web = score_resumes_batch(SAMPLE_JOB_DESCRIPTIONS["Full-Stack Web Developer"], SAMPLE_RESUMES)
fig_cmp = plot_candidate_comparison(res_web[0], res_web[1])
assert fig_cmp is not None, "Comparison chart failed"
print(f"   [PASS] Head-to-Head Comparison chart rendered for {res_web[0]['candidate_name']} vs {res_web[1]['candidate_name']}!")

print("\n7. Testing Printable HTML Evaluation Report generation...")
html_out = generate_html_report(res_web, meta_web, 60)
assert "Candidate Screening & Evaluation Report" in html_out, "HTML generation failed"
print(f"   [PASS] Formatted HTML evaluation report generated ({len(html_out)} bytes)!")

print("\n8. Testing Plotly dashboard visualizations...")
fig1 = plot_leaderboard(res_web)
fig2 = plot_score_breakdown(res_web)
fig3 = plot_skill_radar(res_web[0]["category_breakdown"])
fig4 = plot_score_gauge(res_web[0]["composite_score"])
print("   [PASS] All 4 dashboard Plotly charts initialized without errors!")

print("\n9. Testing Personalized Interview Question Generator...")
guide = generate_candidate_interview_guide(res_web[0], meta_web)
assert len(guide["matched_questions"]) > 0 or len(guide["missing_questions"]) > 0, "Interview guide empty"
assert len(guide["behavioral_questions"]) > 0, "Behavioral questions missing"
print(f"   [PASS] Generated interview guide with {len(guide['matched_questions'])} technical and {len(guide['behavioral_questions'])} behavioral questions!")

print("\nALL SYSTEM TESTS PASSED SUCCESSFULLY! ZERO ERRORS DETECTED.")
