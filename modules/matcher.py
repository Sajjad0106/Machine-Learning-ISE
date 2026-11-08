"""
Core Machine Learning and Natural Language Processing Matching Engine.
Implements TF-IDF Vector Space Model, Cosine Similarity, Multi-Category Skill Matching,
and a Multi-Criteria Weighted Decision Score for Resume Screening.
"""

import re
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from modules.skills_db import ALL_SKILLS, get_skill_category
from modules.extractor import (
    extract_contact_info,
    extract_candidate_name,
    extract_education_level,
    extract_experience_years,
    audit_resume_structure,
)


def normalize_text(text: str) -> str:
    """Normalize text while carefully preserving programming language tokens (c++, c#, .net, etc.)."""
    text = text.lower()
    # Normalize unicode spaces and hyphens
    text = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015]', '-', text)
    # Preserve key symbols in technical terms
    text = re.sub(r'(?<=[a-z0-9])\.(?=[a-z0-9])', '.', text)
    text = re.sub(r'[^a-z0-9+#.\s/-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_skills_from_text(text: str, skill_pool: List[str] = None) -> List[str]:
    """Find all matching skills in the provided text using boundary-safe regex matching."""
    if skill_pool is None:
        skill_pool = ALL_SKILLS

    norm_text = f" {normalize_text(text)} "
    detected = []

    for skill in skill_pool:
        s_norm = normalize_text(skill)
        if not s_norm:
            continue
        # Escape special regex chars like ++ or #
        pattern = r'(?<![a-z0-9])' + re.escape(s_norm) + r'(?![a-z0-9])'
        if re.search(pattern, norm_text):
            detected.append(skill)

    return sorted(set(detected), key=str.lower)


def extract_required_experience_from_jd(jd_text: str) -> float:
    """Detect minimum required experience specified in the job description."""
    matches = re.findall(r'(\d+(?:\.\d+)?)\+?\s*(?:-\s*\d+\s*)?(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)', jd_text, re.IGNORECASE)
    if matches:
        try:
            return min(float(m) for m in matches)
        except ValueError:
            pass
    return 0.0


def score_resumes_batch(
    job_description: str,
    resumes: Dict[str, str],
    custom_skills: List[str] = None,
    weights: Dict[str, float] = None,
    anonymize: bool = False
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Score a batch of resumes against a job description.
    
    Args:
        job_description: Raw JD text.
        resumes: Dictionary of {filename: raw_resume_text}.
        custom_skills: Additional user-defined skills.
        weights: Dictionary with weights for 'skills', 'semantic', 'experience', 'education'.
        anonymize: Whether to mask candidate identities for blind review.
        
    Returns:
        Tuple of (ranked_results_list, jd_analysis_dict)
    """
    if weights is None:
        weights = {"skills": 0.45, "semantic": 0.35, "experience": 0.10, "education": 0.10}

    # Normalize weights to sum to 1.0
    total_w = sum(weights.values()) or 1.0
    w_skills = weights.get("skills", 0.45) / total_w
    w_semantic = weights.get("semantic", 0.35) / total_w
    w_exp = weights.get("experience", 0.10) / total_w
    w_edu = weights.get("education", 0.10) / total_w

    # Skill dictionary setup
    combined_skills = list(ALL_SKILLS)
    if custom_skills:
        combined_skills.extend([s.strip().lower() for s in custom_skills if s.strip()])
    combined_skills = sorted(set(combined_skills), key=str.lower)

    # 1. Analyze Job Description
    jd_skills = extract_skills_from_text(job_description, combined_skills)
    jd_required_exp = extract_required_experience_from_jd(job_description)
    jd_edu = extract_education_level(job_description)

    # Group JD skills by category
    jd_skills_by_cat = {}
    for sk in jd_skills:
        cat = get_skill_category(sk)
        jd_skills_by_cat.setdefault(cat, []).append(sk)

    # 2. Vector Space Model (TF-IDF + Cosine Similarity)
    filenames = list(resumes.keys())
    resume_texts = list(resumes.values())
    corpus = [job_description] + resume_texts

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        max_df=0.95,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # 3. Evaluate each resume individually
    results = []
    for idx, (filename, resume_text) in enumerate(zip(filenames, resume_texts)):
        # Cosine similarity between asymmetric docs (JD vs Resume) typically peaks around 0.35 - 0.45.
        # We report raw cosine percentage and an empirically calibrated semantic relevance percentage.
        semantic_sim = float(similarities[idx])
        raw_cosine = round(semantic_sim * 100, 1)
        semantic_score = round(min(100.0, (semantic_sim / 0.35) * 100.0), 1)

        # Candidate Info & Entities
        contact = extract_contact_info(resume_text)
        candidate_name = extract_candidate_name(resume_text, filename)
        candidate_edu = extract_education_level(resume_text)
        candidate_exp = extract_experience_years(resume_text)
        resume_audit = audit_resume_structure(resume_text)

        # Skill matching
        candidate_skills = extract_skills_from_text(resume_text, combined_skills)
        matched_skills = sorted(set(jd_skills) & set(candidate_skills), key=str.lower)
        missing_skills = sorted(set(jd_skills) - set(candidate_skills), key=str.lower)
        additional_skills = sorted(set(candidate_skills) - set(jd_skills), key=str.lower)

        # Skill coverage calculation
        if jd_skills:
            skill_coverage = len(matched_skills) / len(jd_skills)
        else:
            skill_coverage = 1.0 if candidate_skills else 0.5
        skill_score = round(skill_coverage * 100, 1)

        # Experience score calculation
        if jd_required_exp > 0:
            exp_ratio = min(candidate_exp / jd_required_exp, 1.5)
            # Full points if candidate meets or exceeds required experience
            exp_score = min(exp_ratio * 100, 100.0)
        else:
            # Baseline based on experience presence
            exp_score = min(candidate_exp * 20.0, 100.0) if candidate_exp > 0 else 70.0
        exp_score = round(exp_score, 1)

        # Education score calculation
        edu_rank = {
            "PhD / Doctorate": 100,
            "Master's Degree": 90,
            "Bachelor's Degree": 80,
            "Associate's / Diploma": 65,
            "Not Specified": 50
        }
        edu_score = edu_rank.get(candidate_edu["highest_level"], 60)

        # Composite Relevance Score Calculation
        composite_score = (
            (w_skills * skill_score) +
            (w_semantic * semantic_score) +
            (w_exp * exp_score) +
            (w_edu * edu_score)
        )
        composite_score = round(float(np.clip(composite_score, 0, 100)), 1)

        # Category-wise skill breakdown
        cat_breakdown = {}
        for cat in set(list(jd_skills_by_cat.keys()) + ["Other / Custom"]):
            jd_in_cat = [s for s in jd_skills if get_skill_category(s) == cat]
            cand_in_cat = [s for s in candidate_skills if get_skill_category(s) == cat]
            matched_in_cat = [s for s in matched_skills if get_skill_category(s) == cat]
            if jd_in_cat or cand_in_cat:
                cat_breakdown[cat] = {
                    "jd_count": len(jd_in_cat),
                    "cand_count": len(cand_in_cat),
                    "matched_count": len(matched_in_cat),
                    "matched_skills": matched_in_cat
                }

        # Recommendation Tier
        if composite_score >= 80:
            tier = "Strong Shortlist (Tier 1)"
            tier_color = "green"
        elif composite_score >= 65:
            tier = "Potential Match (Tier 2)"
            tier_color = "blue"
        elif composite_score >= 50:
            tier = "Moderate Match (Tier 3)"
            tier_color = "orange"
        else:
            tier = "Low Match / Needs Review"
            tier_color = "red"

        # Anonymization for Blind Screening
        display_name = f"Candidate #{idx + 1}" if anonymize else candidate_name
        display_contact = {
            "email": "[Protected for Blind Review]" if anonymize else contact["email"],
            "phone": "[Protected for Blind Review]" if anonymize else contact["phone"],
            "linkedin": "[Protected for Blind Review]" if anonymize else contact["linkedin"],
            "github": contact["github"],
            "portfolio": contact["portfolio"]
        }

        results.append({
            "rank": 0,  # filled after sort
            "candidate_name": display_name,
            "raw_name": candidate_name,
            "filename": filename,
            "composite_score": composite_score,
            "tier": tier,
            "tier_color": tier_color,
            "scores_breakdown": {
                "skill_score": skill_score,
                "semantic_score": semantic_score,
                "raw_cosine": raw_cosine,
                "exp_score": exp_score,
                "edu_score": edu_score,
            },
            "contact": display_contact,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "additional_skills": additional_skills,
            "candidate_skills": candidate_skills,
            "experience_years": candidate_exp,
            "education": candidate_edu,
            "category_breakdown": cat_breakdown,
            "resume_audit": resume_audit,
            "_raw_text": resume_text
        })

    # Sort candidates by composite score descending
    results = sorted(results, key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    jd_meta = {
        "required_skills": jd_skills,
        "skills_by_category": jd_skills_by_cat,
        "required_experience": jd_required_exp,
        "education": jd_edu,
        "weights_used": {
            "skills": round(w_skills * 100),
            "semantic": round(w_semantic * 100),
            "experience": round(w_exp * 100),
            "education": round(w_edu * 100)
        }
    }

    return results, jd_meta
