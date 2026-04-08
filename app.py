import os
import uvicorn
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.env.environment import ResumeEnv
from app.env.models import Action
from app.matching.matcher import match_easy, match_medium, match_hard, match_random, get_top_k
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
    border-radius: 15px; padding: 15px; font-family: 'JetBrains Mono', monospace; color: #00ffcc; height: 350px; overflow-y: auto; border: 1px solid rgba(0, 255, 204, 0.1);
}

.charts-row { display: flex; flex-direction: column; gap: 12px; }
.chart-container { height: 180px; position: relative; }

.leaderboard-entry { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }
.leaderboard-entry b { font-size: 1.1rem; }

.modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); backdrop-filter: blur(10px); justify-content: center; align-items: center; }
.modal-content { background: #0e0e1a; border: 1px solid #4facfe; border-radius: 24px; padding: 25px; width: 85%; max-height: 85vh; overflow-y: auto; }

.badge { background: rgba(79, 172, 254, 0.15); color: #4facfe; border: 1px solid rgba(79, 172, 254, 0.3); padding: 5px 10px; border-radius: 8px; font-size: 0.75rem; margin-right: 6px; display: inline-block; margin-bottom: 6px; }

.trust-meter {
    height: 10px; background: rgba(255,255,255,0.05); border-radius: 5px; overflow: hidden; margin-top: 10px;
}
.trust-fill {
    height: 100%; background: linear-gradient(90deg, #fe4a90, #00ffcc); width: 100%; transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}
"""

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OpenEnv Arena v4.0 | Multi-Step HR</title>
    <style>REPLACE_CSS</style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>

<div class="glass-container">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <h1>🌌 OpenEnv Arena v4.0</h1>
        <div style="background:rgba(255,255,255,0.05); padding:6px 12px; border-radius:10px; font-size:0.8rem;">Architecture: <b>Stateful Sequential</b></div>
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
                <h3 style="margin-top:0; color:#4facfe;">📊 Simulation Stats</h3>
                <div class="leaderboard-entry"><span>Trust Multiplier</span> <b id="ui-trust" style="color:#00ffcc;">1.00</b></div>
                <div class="trust-meter"><div id="trust-fill" class="trust-fill" style="width: 100%;"></div></div>
                <div class="leaderboard-entry" style="margin-top:15px;"><span>Shortlisted</span> <b id="ui-shortlist" style="color:#4facfe;">0</b></div>
            </div>
        </div>

        <div class="col-mid">
            <div class="terminal" id="terminal">
                <div style="color:#888;">>> Deep HR Reasoning Engine Ready...</div>
            </div>
            <div class="charts-row">
                <div class="chart-container"><canvas id="rewardChart"></canvas></div>
                <div class="chart-container"><canvas id="skillRadar"></canvas></div>
            </div>
        </div>

        <div class="col-right">
            <div class="task-card" style="height:100%; border-color: rgba(79, 172, 254, 0.2);">
                <h3 style="margin-top:0;">📡 Decision Analytics</h3>
                <div style="margin-bottom:20px;">
                    <div style="color:#a0a0b0; font-size:0.85rem;">REWARD SCORE</div>
                    <div id="xai-score" style="font-size:2.8rem; font-weight:bold; color:#00ffcc;">0.00</div>
                </div>
                <div>
                    <div style="color:#a0a0b0; font-size:0.85rem;">SKILL ALIGNMENT</div>
                    <div id="xai-matched" style="margin-top:8px;">--</div>
                </div>
                <div style="margin-top:20px;">
                    <div style="color:#a0a0b0; font-size:0.85rem;">ADAPTIVE FEEDBACK</div>
                    <div id="xai-sugg" style="color:#fe4a90; margin-top:8px; font-weight:600; line-height:1.4;">Simulation Standby...</div>
                </div>
            </div>
        </div>
    </div>
</div>

<div id="inspector-modal" class="modal">
    <div class="modal-content">
        <span onclick="closeModal()" style="float:right; cursor:pointer; font-size:2rem;">&times;</span>
        <h2 style="color:#4facfe;">🔍 Multi-Step Inspector</h2>
        <div id="modal-body"></div>
    </div>
</div>

<script>
    const term = document.getElementById('terminal');
    let rewardChart, skillRadar;

    function initCharts() {
        rewardChart = new Chart(document.getElementById('rewardChart'), {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Reward trajectory', data: [], borderColor: '#fe4a90', tension: 0.3, fill: true, backgroundColor: 'rgba(254, 74, 144, 0.1)' }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0, max: 1 } } }
        });
        skillRadar = new Chart(document.getElementById('skillRadar'), {
            type: 'radar',
            data: { labels: ['Matched', 'Missing', 'Complexity', 'Diversity', 'Alignment'], 
                   datasets: [{ label: 'HR Compliance', data: [0, 0, 0, 0, 0], backgroundColor: 'rgba(0, 255, 204, 0.2)', borderColor: '#00ffcc', pointBackgroundColor: '#00ffcc' }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { r: { suggestedMin: 0, suggestedMax: 10 } } }
        });
    }

    function log(msg, type='info', data=null) {
        const div = document.createElement('div');
        div.style.padding = '4px 0';
        div.style.color = type === 'reasoning' ? '#4facfe' : (type === 'success' ? '#00ffcc' : (type === 'step' ? '#fe4a90' : '#fff'));
        div.innerHTML = msg;
        if (data) { div.onclick = () => openInspector(data); div.style.cursor = 'pointer'; div.title = "Click to inspect step details"; }
        term.appendChild(div);
        term.scrollTop = term.scrollHeight;
    }

    function openInspector(data) {
        document.getElementById('modal-body').innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        document.getElementById('inspector-modal').style.display = 'flex';
    }
    function closeModal() { document.getElementById('inspector-modal').style.display = 'none'; }

    async function runTask(taskType, btn) {
        btn.disabled = true;
        term.innerHTML = ''; // Clear terminal
        log(`>> ARCHITECTURE: OpenEnv Stateful Simulation Initiated...`, 'step');
        
        try {
            const res = await fetch(`/api/run/${taskType}`);
            const data = await res.json();
            
            for (let entry of data.logs) {
                await new Promise(r => setTimeout(r, 600));
                log(entry.msg, entry.status, entry.data);
                
                // Update Trust UI
                if (entry.trust !== undefined) {
                    document.getElementById('ui-trust').innerText = entry.trust.toFixed(2);
                    document.getElementById('trust-fill').style.width = `${(entry.trust - 0.5) / 0.5 * 100}%`;
                }
                
                // Update Shortlist UI
                if (entry.shortlist_count !== undefined) {
                    document.getElementById('ui-shortlist').innerText = entry.shortlist_count;
                }

                if (entry.reward !== undefined) {
                    if (rewardChart.data.labels.length > 50) {
                        rewardChart.data.labels.shift();
                        rewardChart.data.datasets[0].data.shift();
                    }
                    rewardChart.data.labels.push(`L${rewardChart.data.labels.length + 1}`);
                    rewardChart.data.datasets[0].data.push(entry.reward);
                    rewardChart.update();
                }

                if (entry.xai) {
                    document.getElementById('xai-matched').innerHTML = entry.xai.matched_skills.map(s => `<span class="badge">${s}</span>`).join('') || '--';
                    document.getElementById('xai-sugg').innerText = entry.xai.suggestion || 'Analyzing...';
                    
                    skillRadar.data.datasets[0].data = [
                        entry.xai.matched_skills.length, 
                        entry.xai.missing_skills.length, 
                        6, 4, entry.reward * 10
                    ];
                    skillRadar.update();
                    document.getElementById('xai-score').innerText = entry.reward.toFixed(2);
                }
            }
            
        } catch(e) { log(">> CRITICAL ERROR: Environment simulation interrupted.", 'step'); }
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
    
    all_resumes_dict = [r.model_dump() for r in obs.resumes]
    all_jobs_dict = [j.model_dump() for j in obs.jobs]
    
    # ---------------------------------------------------------
    # 🚀 STEP 1: ANALYZE_JOB
    # ---------------------------------------------------------
    obs, reward, done, _ = env.step(Action(action_type="analyze_job"))
    logs.append({
        "msg": f">> STEP 1: [Analyze] Extracting keyword dependencies for {task_type.upper()} task.",
        "status": "step",
        "reward": reward.score,
        "trust": reward.trust_score,
        "data": {"jobs": all_jobs_dict}
    })

    # ---------------------------------------------------------
    # 🚀 STEP 2: SHORTLIST
    # ---------------------------------------------------------
    primary_job = all_jobs_dict[0]
    shortlist_ids = get_top_k(primary_job, all_resumes_dict, k=5)
    obs, reward, done, _ = env.step(Action(action_type="shortlist", resumes=shortlist_ids))
    logs.append({
        "msg": f">> STEP 2: [Shortlist] Identifed top-5 candidates via semantic similarity.",
        "status": "step",
        "reward": reward.score,
        "trust": reward.trust_score,
        "shortlist_count": len(obs.shortlisted_resumes),
        "data": {"shortlisted_ids": shortlist_ids}
    })

    # ---------------------------------------------------------
    # 🚀 STEP 3: RANK
    # ---------------------------------------------------------
    obs, reward, done, _ = env.step(Action(action_type="rank"))
    logs.append({
        "msg": f">> STEP 3: [Rank] Calculating optimal decision path across all available vectors.",
        "status": "step",
        "reward": reward.score,
        "trust": reward.trust_score,
        "data": {"current_matches": obs.current_matches}
    })

    # ---------------------------------------------------------
    # 🚀 STEP 4: FINALIZE
    # ---------------------------------------------------------
    if task_type == "easy":
        matches = match_easy(all_resumes_dict, all_jobs_dict)
        final_act = Action(action_type="finalize", matches=matches)
    elif task_type == "medium":
        ranked = match_medium(all_resumes_dict, all_jobs_dict)
        final_act = Action(action_type="finalize", ranked_list=ranked)
    else: # hard
        matches = match_hard(all_resumes_dict, all_jobs_dict)
        final_act = Action(action_type="finalize", matches=matches)

    obs, reward, done, _ = env.step(final_act)
    logs.append({
        "msg": f">> STEP 4: [Finalize] Optimal Batch Assignment applied successfully.",
        "status": "success",
        "reward": reward.score,
        "trust": reward.trust_score,
        "xai": reward.model_dump(),
        "data": {"final_matches": obs.current_matches}
    })
        
    return {"logs": logs}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
