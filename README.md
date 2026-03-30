# Badminton Tracking & Analysis

A practical badminton video analysis pipeline focused on **stable 2-player tracking + shuttle tracking + pose visualization**.

[![Replay3D CI](https://github.com/WilliamK112/badminton-ai/actions/workflows/replay3d-ci.yml/badge.svg)](https://github.com/WilliamK112/badminton-ai/actions/workflows/replay3d-ci.yml)

> GitHub Actions CI: see `.github/workflows/replay3d-ci.yml` for the Replay3D package+gate regression workflow.
>
> Replay3D package artifact download (for PR review):
> 1. Open the workflow run: <https://github.com/WilliamK112/badminton-ai/actions/workflows/replay3d-ci.yml>
> 2. In the run page, open the **Summary** tab to see the Replay3D Job Summary block (now sourced from `runs/replay3d_full_gate_summary.md`, combining full-gate/evidence/tuning pass markers + latest package artifact sizes).
> 3. Scroll to **Artifacts** and download `replay3d-ci-package-<run_id>` (retained for 14 days).
> 4. Unzip and inspect package files (`replay_3d.jsonl`, `preview/`, `quality/report.md`, `replay_3d_meta.json`, `manifest.md`).
>
> Quick UI map (text anchor for reviewers): on each workflow run page, the replay summary is in the center **Summary** tab under step **Build replay3d full-gate summary artifact**; package download is in the bottom-right **Artifacts** panel.

## Demo (Latest)

<table>
  <tr>
    <td><img src="./demo/gif/%E6%88%AA%E5%B1%8F2026-03-25%2001.40.31.png" alt="screenshot" width="420"/></td>
    <td><img src="./demo/gif/frame_0010.jpg" alt="frame_0010" width="420"/></td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <img src="./demo/gif/badminton_github_long.gif" alt="badminton_github_long" width="900"/>
    </td>
  </tr>
</table>

## About

This project is built for demo-ready, reproducible badminton analysis outputs from match videos.
Current focus:
- Keep only on-court 2 players (avoid audience / sideline jumps)
- Keep shuttle trajectory continuous and visible
- Preserve human pose (skeleton) data and visualization in the final output

## Tech Stack / Models Used

### Core Libraries
- Python
- OpenCV
- Ultralytics YOLO
- NumPy / Pandas / SciPy

### Detection & Tracking Models
- **Player detection/tracking:** `yolo11n.pt` (person class + temporal constraints)
- **Pose estimation:** `yolo11n-pose.pt`
- **Shuttle primary model:**
  - `Badminton-Analysis/train/shuttle_output/models/weights/best.pt`
- **Shuttle fallback model:** `yolo11n.pt` (sports-ball class fallback)

### Tracking/Filtering Strategies
- ROI filtering (court-focused region)
- Top/Bottom 2-player slot assignment
- Temporal anti-jump constraints for player boxes
- Shuttle missing-frame recovery via interpolation + velocity-based prediction

## Project Structure

```text
badminton-ai/
├── README.md
├── PROJECT_STATUS.md
├── src/
│   ├── pipeline/
│   │   └── runner.py                # main entry
│   ├── integration/
│   │   └── pipeline.py              # player + shuttle + pose integration
│   ├── player/
│   │   └── tracker.py               # 2-player constrained tracker
│   ├── shuttle/
│   │   └── tracker.py               # rewritten shuttle tracker (primary+fallback)
│   └── ...                          # historical experiments / training scripts
├── demo/
│   └── gif/
│       ├── 截屏2026-03-25 01.40.31.png
│       ├── frame_0010.jpg
│       └── badminton_github_long.gif
└── output_*/                         # local generated outputs (ignored in git)
```

## Current Output Style

Output video includes:
- Player 1 / Player 2 labels (on top of boxes)
- Skeleton keypoints + limbs
- Shuttle marker + `SHUTTLE` label

## Quick Start

```bash
cd /Users/William/.openclaw/workspace/projects/badminton-ai
source .venv/bin/activate

python -m src.pipeline.runner thisone.mp4 -o output_demo -i 2 -m 3000
```

## Replay3D Quick Commands

Use the project-local replay venv by default (`PY_REPLAY3D=./.venv-replay3d/bin/python`).

```bash
# Build one validated Replay3D package (JSONL + preview + quality + meta + manifest)
make replay3d-package

# Run package gate smoke (healthy package must pass, corrupted package must fail)
make replay3d-gate

# Local PR preflight (replay3d-ci + replay3d-inspect-ci)
make replay3d-pr-check

# CI-style one-command regression (package + gate smoke)
make replay3d-ci

# Inspect the latest CI package locally (required files + size)
make replay3d-inspect

# CI-friendly inspection gate (non-zero exit if missing/zero-size required files)
make replay3d-inspect-ci

# Deterministic strict-fallback regression (newest partial package should fallback to latest complete package)
make replay3d-inspect-fallback-regression

# Collect PR-ready CI evidence markdown (latest package path + pass markers + artifact sizes)
make replay3d-evidence

# Strict evidence gate (non-zero if markers/rows are incomplete)
make replay3d-evidence-strict

# One-command local PR + evidence preflight
make replay3d-pr-evidence-check

# Generate tuning comparison report (default vs smoother) with recommendation
make replay3d-tuning-report

# Generate one-screen tuning summary (recommended params + before/after metrics)
make replay3d-tuning-summary

# Generate copy-paste-ready PR comment block from tuning summary
make replay3d-tuning-pr-comment

# Validate PR comment contains required markers/rows (CI-gate friendly)
make replay3d-tuning-pr-comment-check

# Final local release gate (PR evidence gate + tuning PR-comment gate)
make replay3d-full-gate

# One-screen full-gate summary for PR/CI paste
make replay3d-full-gate-summary

# Compact reviewer snippet extracted from full-gate summary
make replay3d-full-gate-snippet

# Validate reviewer snippet required PASS markers (CI/local shared gate)
make replay3d-full-gate-snippet-check

# Verify workflow gate-step ordering proof (strict->check->summary+snippet->snippet-check->summary-write)
make replay3d-workflow-order-check

# Regression-check that full-gate summary always includes workflow proof block + ordered markers
make replay3d-summary-proof-check

# Emit one-glance pointer markdown for latest package/evidence/summary/snippet paths
make replay3d-artifact-pointer

# Validate artifact pointer markdown has required path markers (CI/local gate)
make replay3d-artifact-pointer-check

# Collect evidence for a pinned historical package
make replay3d-evidence-run RUN_DIR=runs/replay3d_ci_20260329_044419_package OUTPUT=runs/replay3d_ci_evidence_20260329_044419.md

# Fail fast if an evidence markdown is missing required pass markers
make replay3d-evidence-check OUTPUT=runs/replay3d_ci_evidence_latest.md

# (Optional) Generate evidence for a pinned historical package run dir
./scripts/replay3d_collect_ci_evidence.sh \
  --run-dir runs/replay3d_ci_20260329_044419_package \
  --output runs/replay3d_ci_evidence_20260329_044419.md
```

Tip: override Python explicitly when needed:

```bash
make replay3d-ci PY_REPLAY3D=./.venv-replay3d/bin/python
```

Review a historical package (non-latest) with pinned run dir evidence:

```bash
./scripts/replay3d_collect_ci_evidence.sh \
  --run-dir runs/replay3d_ci_20260329_044419_package \
  --output runs/replay3d_ci_evidence_pinned.md
```

## Notes

- This repo keeps the latest demo-facing pipeline and visuals clean.
- Large temporary artifacts (outputs/reports/trash) are local and excluded from git.
