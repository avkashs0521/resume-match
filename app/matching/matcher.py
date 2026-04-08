from .similarity import compute_similarity
import numpy as np
from scipy.optimize import linear_sum_assignment


# ✅ EASY → best match only
def match_easy(resumes, jobs):
    sim = compute_similarity(resumes, jobs)

    matches = {}
    for i, job in enumerate(jobs):
        best_idx = np.argmax(sim[i])
        matches[job["id"]] = resumes[best_idx]["id"]

    return matches


# ✅ MEDIUM → ranking with filtering
def match_medium(resumes, jobs):
    job = jobs[0]

    sim = compute_similarity(resumes, [job])[0]

    # 🔥 SKILL BOOST
    boosted_scores = []

    for i, r in enumerate(resumes):
        text = str(r.get("text", "")).lower()
        skills = job.get("skills_required", [])

        skill_match = sum(skill.lower() in text for skill in skills)

        # boost score
        score = sim[i] + 0.4 * skill_match
        boosted_scores.append(score)

    ranked_indices = np.argsort(boosted_scores)[::-1]

    ranked_resumes = [resumes[i]["id"] for i in ranked_indices[:3]]

    return ranked_resumes

# ✅ HARD → optimal assignment
def match_hard(resumes, jobs):
    sim = compute_similarity(resumes, jobs)

    cost = -sim
    row_ind, col_ind = linear_sum_assignment(cost)

    assignments = {}
    for j, r in zip(row_ind, col_ind):
        assignments[jobs[j]["id"]] = resumes[r]["id"]

    return assignments


# ✅ RANDOM → baseline (for comparison)
def match_random(resumes, jobs, task_type="easy"):
    import random
    r_ids = [r["id"] for r in resumes]
    j_ids = [j["id"] for j in jobs]
    
    if task_type == "medium":
        return random.sample(r_ids, min(3, len(r_ids)))
    
    # For easy and hard, assign one resume per job randomly
    matches = {}
    for jid in j_ids:
        matches[jid] = random.choice(r_ids)
    return matches