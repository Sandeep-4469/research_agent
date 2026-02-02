# SHM Research Assistant (Google ADK)

This project implements a **Sequential LLM-based research assistant** for  
**Structural Health Monitoring (SHM) using AI**, built with **Google Agent Development Kit (ADK)**.

## What it does
The agent pipeline automatically:
- Finds public SHM datasets
- Identifies AI/ML codebases
- Lists relevant research papers
- Verifies authenticity
- Generates an academic-style survey report

## How it works
The system uses multiple **LLM-based agents** executed sequentially:

Each agent performs one task and passes its output to the next agent via shared state.

> This system uses **LLM agents (Google Gemini)** and **does NOT use reinforcement learning**.

## How to run
- in root folder run `adk web`
- Select `agents`
- Click **Run** and give the specific topic on which you want find datasets and papers


## Credits
Built using **Google Agent Development Kit (ADK)**  
© Google LLC
