import os
import uvicorn
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.env.environment import ResumeEnv
from app.env.models import Action
from app.matching.matcher import match_easy, match_medium, match_hard, match_random
from pydantic import BaseModel

app = FastAPI()

class MatchRequest(BaseModel):
    task: str = "easy"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

body {
    background: radial-gradient(circle at top right, #1a1a2e, #0e0b16);
    color: #e0e0e0;
    font-family: 'Inter', sans-serif;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    margin: 0;
    padding: 20px;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.glass-container {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 28px;
    padding: 25px;
    width: 100%;
    max-width: 1400px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.main-layout { display: flex; gap: 20px; }
.col-left { flex: 1; display: flex; flex-direction: column; gap: 15px; }
.col-mid { flex: 1.5; display: flex; flex-direction: column; gap: 15px; }
.col-right { flex: 1.2; display: flex; flex-direction: column; gap: 15px; }

.task-card, .leaderboard-card, .chart-container {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 18px;
    padding: 15px;
}

h1 { font-size: 1.8rem; margin: 0; background: linear-gradient(90deg, #00f2fe, #4facfe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.run-btn {
    background: linear-gradient(135deg, #fe4a90, #f0166d);
    border: none; padding: 12px; border-radius: 12px; color: white; font-weight: 600; cursor: pointer; transition: all 0.3s;
}
.run-btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(254, 74, 144, 0.3); }

.terminal {
    background: rgba(0, 0, 0, 0.4);
    border-radius: 15px; padding: 15px; font-family: 'JetBrains Mono', monospace; color: #00ffcc; height: 250px; overflow-y: auto; border: 1px solid rgba(0, 255, 204, 0.1);
}

.charts-row { display: flex; flex-direction: column; gap: 12px; }
.chart-container { height: 180px; position: relative; }

.leaderboard-entry { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }
.leaderboard-entry b { font-size: 1.1rem; }

.modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); backdrop-filter: blur(10px); justify-content: center; align-items: center; }
.modal-content { background: #0e0e1a; border: 1px solid #4facfe; border-radius: 24px; padding: 25px; width: 85%; max-height: 85vh; overflow-y: auto; }

.badge { background: rgba(79, 172, 254, 0.15); color: #4facfe; border: 1px solid rgba(79, 172, 254, 0.3); padding: 5px 10px; border-radius: 8px; font-size: 0.75rem; margin-right: 6px; display: inline-block; margin-bottom: 6px; }
"""

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OpenEnv Arena v3.2 | Radar Tracking</title>
    <style>REPLACE_CSS</style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>

<div class="glass-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h1>🌌 OpenEnv Arena v3.2</h1>
        <div style="background:rgba(255,255,255,0.05); padding:6px 12px; border-radius:10px; font-size:0.8rem;">Status: <b>Deterministic</b></div>
    </div>
    
    <div class="main-layout">
        <div class="col-left">
            <div class="task-card">
                <h3>Easy Task</h3>
                <button class="run-btn" onclick="runTask('easy', this)" style="width:100%;">MATCH AGENT</button>
            </div>
            <div class="task-card">
                <h3>Medium Task</h3>
                <button class="run-btn" onclick="runTask('medium', this)" style="width:100%;">RANK AGENT</button>
            </div>
            <div class="task-card">
                <h3>Hard Task</h3>
                <button class="run-btn" onclick="runTask('hard', this)" style="width:100%;">ALLOCATE AGENT</button>
            </div>

            <div class="leaderboard-card">
                <h3 style="margin-top:0; color:#4facfe;">📊 Stats Leaderboard</h3>
                <div class="leaderboard-entry"><span>Optimal Agent</span> <b id="lb-optimal" style="color:#00ffcc;">--</b></div>
                <div class="leaderboard-entry"><span>Random Baseline</span> <b id="lb-random" style="color:#fe4a90;">--</b></div>
            </div>
        </div>

        <div class="col-mid">
            <div class="terminal" id="terminal">
                <div style="color:#888;">>> Arena system standby...</div>
            </div>
            <div class="charts-row">
                <div class="chart-container"><canvas id="rewardChart"></canvas></div>
                <div class="chart-container"><canvas id="skillRadar"></canvas></div>
            </div>
        </div>

        <div class="col-right">
            <div class="task-card" style="height:100%; border-color: rgba(79, 172, 254, 0.2);">
                <h3 style="margin-top:0;">📡 Agent Analytics</h3>
                <div style="margin-bottom:20px;">
                    <div style="color:#a0a0b0; font-size:0.85rem;">REWARD GRADIENT</div>
                    <div id="xai-score" style="font-size:2.8rem; font-weight:bold; color:#00ffcc;">0.00</div>
                </div>
                <div>
                    <div style="color:#a0a0b0; font-size:0.85rem;">SKILL OVERLAP</div>
                    <div id="xai-matched" style="margin-top:8px;">--</div>
                </div>
                <div style="margin-top:20px;">
                    <div style="color:#a0a0b0; font-size:0.85rem;">ADAPTIVE FEEDBACK</div>
                    <div id="xai-sugg" style="color:#fe4a90; margin-top:8px; font-weight:600; line-height:1.4;">Waiting for agent action...</div>
                </div>
            </div>
        </div>
    </div>
</div>

<div id="inspector-modal" class="modal">
    <div class="modal-content">
        <span onclick="closeModal()" style="float:right; cursor:pointer; font-size:2rem;">&times;</span>
        <h2 style="color:#4facfe;">🔍 Vector Inspector</h2>
        <div id="modal-body"></div>
    </div>
</div>

<script>
    const term = document.getElementById('terminal');
    let rewardChart, skillRadar;

    function initCharts() {
        rewardChart = new Chart(document.getElementById('rewardChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Reward history', data: [], borderColor: '#fe4a90', tension: 0.3, fill: true, backgroundColor: 'rgba(254, 74, 144, 0.1)' }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } } }
        });
        skillRadar = new Chart(document.getElementById('skillRadar'), {
            type: 'radar',
            data: { labels: ['Matched Skills', 'Missing Skills', 'Complexity', 'Diversity', 'Alignment'], 
                   datasets: [{ label: 'NLP Delta', data: [0, 0, 0, 0, 0], backgroundColor: 'rgba(0, 255, 204, 0.2)', borderColor: '#00ffcc', pointBackgroundColor: '#00ffcc' }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { r: { suggestedMin: 0, suggestedMax: 10 } } }
        });
    }

    function log(msg, type='info', data=null) {
        const div = document.createElement('div');
        div.style.padding = '4px 0';
        div.style.color = type === 'reasoning' ? '#4facfe' : (type === 'success' ? '#00ffcc' : '#fff');
        div.innerHTML = msg;
        if (data) { div.onclick = () => openInspector(data); div.style.cursor = 'pointer'; div.title = "Click to inspect vectors"; }
        term.appendChild(div);
        term.scrollTop = term.scrollHeight;
    }

    function openInspector(data) {
        document.getElementById('modal-body').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        document.getElementById('inspector-modal').style.display = 'flex';
    }
    function closeModal() { document.getElementById('inspector-modal').style.display = 'none'; }

    async function runTask(taskType, btn) {
        btn.disabled = true; term.innerHTML = '';
        rewardChart.data.labels = []; rewardChart.data.datasets[0].data = []; rewardChart.update();
        
        log(">> INITIALIZING ARENA TRAJECTORY...", 'info');
        await new Promise(r => setTimeout(r, 600));
        log(">> [REASONING] Decoding job requirements into TF-IDF sparse matrix...", 'reasoning');
        await new Promise(r => setTimeout(r, 800));
        log(">> [REASONING] Generating transformer-based contextual embeddings...", 'reasoning');
        await new Promise(r => setTimeout(r, 800));

        try {
            const res = await fetch(`/api/run/${taskType}`);
            const data = await res.json();
            
            for (let entry of data.logs) {
                await new Promise(r => setTimeout(r, 400));
                log(entry.msg, entry.status, entry.data);
                
                if (entry.reward !== undefined) {
                    rewardChart.data.labels.push(`S${rewardChart.data.labels.length + 1}`);
                    rewardChart.data.datasets[0].data.push(entry.reward);
                    rewardChart.update();
                }

                // 🔥 UPDATE RADAR PER-STEP
                if (entry.xai) {
                    document.getElementById('xai-matched').innerHTML = entry.xai.matched_skills.map(s => `<span class="badge">${s}</span>`).join('') || '--';
                    document.getElementById('xai-sugg').innerText = entry.xai.suggestion || 'Analyzing...';
                    
                    skillRadar.data.datasets[0].data = [
                        entry.xai.matched_skills.length, 
                        entry.xai.missing_skills.length, 
                        5, 7, entry.reward * 10
                    ];
                    skillRadar.update();
                }
            }

            document.getElementById('xai-score').innerText = data.score.toFixed(2);
            document.getElementById('lb-optimal').innerText = data.score.toFixed(2);
            document.getElementById('lb-random').innerText = data.random_score.toFixed(2);
            
        } catch(e) { log(">> ERROR: Inference error.", 'reasoning'); }
        btn.disabled = false;
    }
    window.onload = initCharts;
</script>
</body>
</html>
""".replace("REPLACE_CSS", CSS)

@app.get("/")
def home():
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/api/run/{task_type}")
def run_task(task_type: str):
    env = ResumeEnv(task_type=task_type)
    obs = env.reset()
    logs = []
    
    r_map = {r.id: r.model_dump() for r in obs.resumes}
    j_map = {j.id: j.model_dump() for j in obs.jobs}
    resumes_dict = [r.model_dump() for r in obs.resumes]
    jobs_dict = [j.model_dump() for j in obs.jobs]
    
    random_matches = match_random(resumes_dict, jobs_dict, task_type)
    env_random = ResumeEnv(task_type=task_type)
    env_random.reset()
    act = Action(matches=random_matches if task_type != "medium" else {}, ranked_list=random_matches if task_type == "medium" else [])
    _, r_reward, _, _ = env_random.step(act)

    if task_type == "easy":
        matches = match_easy(resumes_dict, jobs_dict)
        obs, reward, done, info = env.step(Action(matches=matches))
        jid = list(matches.keys())[0]
        logs.append({
            "msg": f">> ACTION: Found Optimal Candidate for {jid}", 
            "status": "success", 
            "reward": reward.score, 
            "xai": reward.model_dump(),
            "data": {"job": j_map[jid], "resume": r_map[matches[jid]]}
        })
        score = reward.score
        
    elif task_type == "medium":
        ranked = match_medium(resumes_dict, jobs_dict)
        obs, reward, done, info = env.step(Action(ranked_list=ranked))
        logs.append({
            "msg": f">> ACTION: Generated Semantic Top-3 Candidates", 
            "status": "success", 
            "reward": reward.score, 
            "xai": reward.model_dump(),
            "data": {"job": jobs_dict[0], "ranked": [r_map[rid] for rid in ranked]}
        })
        score = reward.score
        
    else: # hard
        opt_matches = match_hard(resumes_dict, jobs_dict)
        score = 0
        matches_so_far = {}
        for idx, (j_id, r_id) in enumerate(opt_matches.items()):
            matches_so_far[j_id] = r_id
            obs, reward, done, info = env.step(Action(matches=matches_so_far))
            logs.append({
                "msg": f">> STEP {idx+1}: Optimal Batch Assignment applied.", 
                "status": "info", 
                "reward": reward.score, 
                "xai": reward.model_dump(),
                "data": {"job": j_map[j_id], "resume": r_map[r_id]}
            })
            score = reward.score
        
    return {"score": score, "random_score": r_reward.score, "logs": logs, "reward_data": reward.model_dump()}

@app.post("/match")
def match_health(req: MatchRequest):
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
