from app.env.environment import ResumeEnv
from app.env.models import Action
from app.matching.matcher import match_easy, match_medium, match_hard
from app.analysis.feedback import generate_feedback


def run():
    print("[START]")

    task_type = "medium"

    env = ResumeEnv(task_type=task_type)
    obs = env.reset()

    resumes = [r.model_dump() for r in obs.resumes]
    jobs = [j.model_dump() for j in obs.jobs]

    if task_type == "easy":
        action = Action(matches=match_easy(resumes, jobs))

    elif task_type == "medium":
        action = Action(ranked_list=match_medium(resumes, jobs))

    elif task_type == "hard":
        action = Action(matches=match_hard(resumes, jobs))

    obs, reward, done, _ = env.step(action)

    print("[STEP]")
    print("Task:", task_type)
    print("Reward:", reward.score)

    if task_type == "medium":
        print("Predicted ranking:", action.ranked_list)

        job_id = list(env.gt.keys())[0]
        print("Ground truth:", env.gt[job_id])

        # 🔥 FEEDBACK BLOCK (NOW CORRECTLY INSIDE)
        job = jobs[0]

        print("\n📊 Resume Feedback:\n")

        for r_id in action.ranked_list:
            resume = next(r for r in resumes if r["id"] == r_id)

            fb = generate_feedback(resume, job)

            print(f"Resume {r_id}:")
            print("  Score:", fb["score"])
            print("  Matched:", fb["matched_skills"])
            print("  Missing:", fb["missing_skills"])
            print("  Suggestion:", fb["suggestion"])
            print()

    print("[END]")


if __name__ == "__main__":
    run()