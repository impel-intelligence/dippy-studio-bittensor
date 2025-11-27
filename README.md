<div align="center">

# Dippy SN11: Enterprise Grade Generative Media Platform on Bittensor  <!-- omit in toc -->


Please check our [Website](https://dippy.studio) for more details on our vision.

[![DIPPY](/banner.png)]([https://dippy.studio))
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---
</div>

- [Introduction](#introduction)
- [Roadmap](#roadmap)
- [Overview of Miner and Validator Functionality](#overview-of-miner-and-validator-functionality)
  - [Miner](#miner)
  - [Validator](#validator)
- [Running Miners and Validators](#running-miners-and-validators)
  - [Running a Miner](#running-a-miner)
  - [Running a Validator](#running-a-validator)
- [License](#license)

---

## Introduction

> **Note:** The following documentation assumes you are familiar with basic Bittensor concepts: Miners, Validators, and incentives. If you need a primer, please check out [Bittensor Docs](https://docs.bittensor.com/learn/bittensor-building-blocks)

Dippy is one of the world's leading AI companion apps with **7M+ users**. The app has ranked [**#3 on the App Store**](https://x.com/angad_ai/status/1850924240742031526) in countries like Germany, been covered by publications like [**Wired magazine**](https://www.wired.com/story/dippy-ai-girlfriend-boyfriend-reasoning/) and the average Dippy user **spends 1+ hour on the app.** 

When Dippy moved beyond text inference, we discovered that Bittensor still doesn’t have production-ready subnets for media inference. The Dippy Studio subnet aims to address this gap in the market and power the fastest, media generation platform on Bittensor. It's scalable, multi-modal, and ready to serve millions of users on Day 1. Our custom, in-house Dippy Inference Engine means that our users can reliably expect the fastest and cheapest inference available on the market.

## Roadmap 

Given the complexity of starting an enterprise grade inference engine from scratch, we plan to divide the process into 3 distinct phases.

**Phase 1:** 
- [ ] Subnet launch with deterministic inference from Day 1
- [ ] Dippy Studio powers all Dippy in-chat images using Dippy Inference Engine serving millions of users
- [ ] Beta version of self-serve API access for external builders to start building

**Phase 2:** 
- [ ] Release dashboard with live statistics on every model
- [ ] Set up our in-house Dippy Inference Engine pipeline to optimize any image model
- [ ] On-board 50+ AI startups with Dippy Studio's image inference

**Phase 3:** 
- [ ] Expand modalities to audio, video, 3D etc while maintaining our speed advantage
- [ ] Power 100+ AI startups with fully decentralized media infrastructure

## Overview of Miner and Validator Functionality

**Miners** Run inference on an optimized version of the image and other media models using Dippy's Inference Engine. 

**Validators** evaluate and assess the output quality, consistency and latency of miner outputs.
Since outputs are deterministic, 

## Running Miners and Validators

### Running a Miner
For detailed miner setup instructions, please see [Miner Documentation](docs/miner.md).

### Running a Validator
For detailed validator setup instructions, please see [Validator Documentation](docs/validator.md).

## License

The Dippy Bittensor subnet is released under the [MIT License](./LICENSE).

# Project Structure Overview

## Core Components

### 1. Miner Implementation
- `dippy-studio-bittensor-miner/` - Main miner application

### 2. Ochestrator Implemetnation
- `dippy-studio-bittensor-orchestrator/` - Orchestrator system

### 3. Documentation
- `docs/` - Project documentation
  - `miner.md` - Miner setup and usage guide
  - `validator.md` - Validator setup and usage guide
