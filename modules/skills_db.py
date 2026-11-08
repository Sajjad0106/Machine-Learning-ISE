"""
Comprehensive Skills Taxonomy for ML-Based Resume Screening System.
Categorized into domains for accurate skill detection and domain gap analysis.
"""

SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "c++", "c#", "c", "javascript", "typescript", "go", "golang",
        "rust", "ruby", "php", "swift", "kotlin", "r", "scala", "dart", "perl", "matlab",
        "shell", "bash", "powershell", "assembly", "lua", "vba"
    ],
    "Machine Learning & AI": [
        "machine learning", "deep learning", "artificial intelligence", "natural language processing",
        "nlp", "computer vision", "large language models", "llm", "transformers", "generative ai",
        "genai", "prompt engineering", "langchain", "llamaindex", "hugging face", "pytorch",
        "tensorflow", "keras", "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
        "opencv", "yolo", "reinforcement learning", "neural networks", "cnn", "rnn", "lstm",
        "bert", "gpt", "rag", "fine-tuning", "diffusion models", "fastai", "spacy", "nltk",
        "gensim", "mediapipe", "speech recognition", "whisper"
    ],
    "Data Science & Analytics": [
        "data analysis", "data science", "pandas", "numpy", "scipy", "statistics",
        "probability", "data visualization", "matplotlib", "seaborn", "plotly", "tableau",
        "power bi", "excel", "advanced excel", "data cleaning", "feature engineering",
        "eda", "exploratory data analysis", "hypothesis testing", "a/b testing", "data mining",
        "business intelligence", "looker", "dbt", "databricks", "apache spark", "pyspark",
        "hadoop", "hive", "kafka", "etl", "data pipelines", "bigquery", "snowflake"
    ],
    "Web & Mobile Development": [
        "html", "html5", "css", "css3", "sass", "tailwind css", "bootstrap", "javascript",
        "react", "react.js", "next.js", "angular", "vue.js", "vue", "svelte", "node.js",
        "express.js", "express", "django", "flask", "fastapi", "spring boot", "asp.net",
        "laravel", "ruby on rails", "graphql", "rest api", "restful api", "websockets",
        "microservices", "frontend", "backend", "full stack", "redux", "zustand",
        "flutter", "react native", "android", "ios", "swiftui", "pwa"
    ],
    "Cloud & DevOps": [
        "cloud computing", "aws", "amazon web services", "azure", "google cloud", "gcp",
        "docker", "kubernetes", "k8s", "ci/cd", "continuous integration", "continuous deployment",
        "jenkins", "github actions", "gitlab ci", "terraform", "ansible", "linux", "ubuntu",
        "helm", "prometheus", "grafana", "nginx", "apache", "serverless", "aws lambda",
        "cloudformation", "vagrant", "openstack", "site reliability engineering", "sre"
    ],
    "Databases & Storage": [
        "sql", "mysql", "postgresql", "postgres", "sqlite", "oracle", "microsoft sql server",
        "mongodb", "redis", "cassandra", "dynamodb", "elasticsearch", "neo4j", "firebase",
        "supabase", "vector databases", "faiss", "chromadb", "pinecone", "qdrant",
        "database design", "indexing", "acid", "orm", "prisma", "sqlalchemy"
    ],
    "Cybersecurity & Networking": [
        "cybersecurity", "information security", "network security", "penetration testing",
        "ethical hacking", "vulnerability assessment", "siem", "soc", "wireshark", "nmap",
        "metasploit", "burp suite", "firewall", "cryptography", "ssl/tls", "owasp",
        "incident response", "malware analysis", "kali linux", "zero trust", "iam",
        "tcp/ip", "dns", "vpn", "routing", "switching"
    ],
    "Software Engineering & Tools": [
        "git", "github", "gitlab", "bitbucket", "jira", "confluence", "trello",
        "data structures", "algorithms", "object-oriented programming", "oop",
        "design patterns", "system design", "agile", "scrum", "kanban", "test-driven development",
        "tdd", "unit testing", "pytest", "jest", "selenium", "postman", "swagger", "openapi",
        "version control", "code review", "debugging"
    ],
    "Soft Skills & Management": [
        "leadership", "communication", "team collaboration", "problem solving",
        "critical thinking", "project management", "time management", "mentoring",
        "stakeholder management", "adaptability", "presentation skills", "work ethic",
        "client communication", "conflict resolution", "creativity"
    ]
}

# Flattened list of all distinct skills
ALL_SKILLS = []
for _cat, _skills in SKILL_TAXONOMY.items():
    ALL_SKILLS.extend(_skills)
ALL_SKILLS = sorted(set(ALL_SKILLS), key=str.lower)


def get_skill_category(skill_name: str) -> str:
    """Find the category for a given skill."""
    norm = skill_name.strip().lower()
    for category, skills in SKILL_TAXONOMY.items():
        if any(norm == s.lower() for s in skills):
            return category
    return "Other / Custom"

