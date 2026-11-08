"""
Interview Question Generator Module for Recruiter Decision Support.
Generates targeted technical and behavioral interview guides based on candidate-specific
skill matches and identified skill gaps.
"""

from typing import Dict, List, Any


DOMAIN_QUESTIONS = {
    "Machine Learning & AI": {
        "matched": [
            ("How do you handle severe class imbalance in real-world tabular and image classification tasks?", 
             "Expected response should cover focal loss, SMOTE, class-weighted cross-entropy, and precision-recall AUC evaluation rather than accuracy."),
            ("Explain the architectural mechanics of the Transformer self-attention mechanism and how multi-head attention improves representation.", 
             "Candidate should explain Q, K, V dot-product scaling, softmax weighting, and parallel sub-space projections."),
            ("What strategies do you employ to prevent data leakage during feature engineering and cross-validation?", 
             "Candidate should explain fitting scalers and encoders strictly on training folds and time-series walk-forward splits.")
        ],
        "missing": [
            ("The role requires production experience with Deep Learning pipelines. How would you design a model retraining and evaluation loop when data distribution drifts?", 
             "Evaluates candidate's conceptual readiness for MLOps, concept drift detection (KS-test/PSI), and automated evaluation.")
        ]
    },
    "Cloud & DevOps": {
        "matched": [
            ("What is the difference between a Kubernetes Deployment, StatefulSet, and DaemonSet, and when would you use each?", 
             "Candidate should distinguish stateless replicas, persistent storage/ordered pod IDs, and node-level operational agents."),
            ("How do you manage Terraform state in a multi-developer team to prevent race conditions and state corruption?", 
             "Candidate should cite remote state backends (e.g. S3 with DynamoDB state locking, or Terraform Cloud)."),
            ("Explain the stages of a robust CI/CD pipeline and how you enforce security gates without slowing down deployments.", 
             "Candidate should mention linting, unit tests, container scanning (Trivy), SAST, and automated canary deployments.")
        ],
        "missing": [
            ("This role requires cloud deployment automation. If you needed to containerize a legacy application with Docker, what steps would you take to minimize image size and attack surface?", 
             "Tests ability to quickly adopt container best practices: multi-stage builds, non-root users, distroless images.")
        ]
    },
    "Web & Mobile Development": {
        "matched": [
            ("How does the browser event loop handle microtasks versus macrotasks in JavaScript / Node.js?", 
             "Candidate should distinguish Promise callbacks (microtasks) from setTimeout/setImmediate (macrotasks) and rendering cycles."),
            ("Explain the trade-offs between Client-Side Rendering (CSR), Server-Side Rendering (SSR), and Static Site Generation (SSG).", 
             "Candidate should explain SEO impact, Time to First Byte (TTFB), First Contentful Paint (FCP), and server overhead."),
            ("How do you optimize slow database queries in an ORM-based backend application?", 
             "Candidate should mention solving N+1 query problems (eager loading), adding composite indexes, and query profiling.")
        ],
        "missing": [
            ("The position requires RESTful API and backend service design. How do you design idempotent endpoints and handle concurrent update conflicts?", 
             "Tests fundamental system design understanding: idempotency keys, HTTP PUT vs POST, and optimistic locking.")
        ]
    },
    "Databases & Storage": {
        "matched": [
            ("Explain the difference between clustered and non-clustered indexes in relational databases, and their impact on write performance.", 
             "Candidate should explain physical row ordering on disk vs B-tree pointer structures and write-amplification overhead."),
            ("When would you choose a NoSQL document database (e.g. MongoDB) over a traditional relational database (e.g. PostgreSQL)?", 
             "Candidate should discuss schema flexibility, horizontal scaling vs ACID transactions and complex join performance.")
        ],
        "missing": [
            ("This role involves relational database modeling. How do you approach database normalization up to 3NF while balancing query read performance?", 
             "Evaluates relational foundation and intentional denormalization strategies for analytics.")
        ]
    },
    "Data Science & Analytics": {
        "matched": [
            ("How do you design and evaluate an A/B test when sample sizes are uneven or metric variance is high?", 
             "Candidate should mention minimum detectable effect (MDE), power calculation, p-values, and bootstrapping."),
            ("Explain how window functions work in SQL and give an example where a window function is superior to a self-join.", 
             "Candidate should demonstrate knowledge of ROW_NUMBER(), RANK(), DENSE_RANK(), and PARTITION BY clauses.")
        ],
        "missing": [
            ("The role requires statistical reporting and business intelligence. How do you translate ambiguous executive questions into measurable KPIs?", 
             "Assesses quantitative problem structuring and business acumen.")
        ]
    },
    "Cybersecurity & Networking": {
        "matched": [
            ("Describe the mechanism of a Cross-Site Request Forgery (CSRF) attack and how modern web frameworks mitigate it.", 
             "Candidate should explain SameSite cookie attributes, anti-CSRF tokens, and double-submit cookies."),
            ("Explain the core principles of a Zero Trust architecture and how it differs from traditional perimeter defense.", 
             "Candidate should cite 'never trust, always verify', least-privilege micro-segmentation, and continuous device/identity validation.")
        ],
        "missing": [
            ("The role requires security vigilance. What baseline controls do you implement to safeguard public-facing APIs from automated credential stuffing?", 
             "Assesses rate limiting, WAF rules, CAPTCHA gating, and abnormal IP velocity detection.")
        ]
    }
}

GENERIC_BEHAVIORAL_QUESTIONS = [
    ("Describe a project where you faced a significant technical roadblock close to a deadline. How did you diagnose the issue and prioritize deliverables?",
     "Look for structured problem-solving, transparent stakeholder communication, and pragmatic engineering trade-offs."),
    ("Tell me about a time you strongly disagreed with a peer's technical architecture decision. How did you resolve the conflict constructively?",
     "Look for data-driven argumentation, active listening, and commitment to the team's final decision."),
    ("How do you stay current with rapidly evolving frameworks and engineering best practices in your technical domain?",
     "Look for genuine curiosity, personal projects, technical reading, and open-source engagement.")
]


def generate_candidate_interview_guide(candidate: Dict[str, Any], jd_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a personalized interview scorecard and question guide based on
    a candidate's matched skills and identified skill gaps.
    """
    matched_skills = candidate.get("matched_skills", [])
    missing_skills = candidate.get("missing_skills", [])
    cat_breakdown = candidate.get("category_breakdown", {})

    matched_questions = []
    missing_questions = []

    # 1. Gather domain-specific questions for matched skills
    for cat, data in cat_breakdown.items():
        if data.get("matched_count", 0) > 0 and cat in DOMAIN_QUESTIONS:
            for q, guide in DOMAIN_QUESTIONS[cat]["matched"]:
                matched_questions.append({
                    "domain": cat,
                    "question": q,
                    "evaluation_guide": guide,
                    "skills_tested": ", ".join(data.get("matched_skills", [])[:4])
                })

    # 2. Gather probing questions for missing skills
    for cat, data in cat_breakdown.items():
        if data.get("jd_count", 0) > 0 and data.get("matched_count", 0) == 0 and cat in DOMAIN_QUESTIONS:
            for q, guide in DOMAIN_QUESTIONS[cat]["missing"]:
                missing_questions.append({
                    "domain": cat,
                    "question": q,
                    "evaluation_guide": guide
                })

    # Limit to top 3-4 questions per section to keep interview practical
    return {
        "candidate_name": candidate.get("candidate_name", "Candidate"),
        "composite_score": candidate.get("composite_score", 0.0),
        "rank": candidate.get("rank", 1),
        "matched_questions": matched_questions[:4],
        "missing_questions": missing_questions[:3],
        "behavioral_questions": GENERIC_BEHAVIORAL_QUESTIONS[:2],
        "primary_strengths": matched_skills[:6],
        "primary_gaps": missing_skills[:6]
    }

