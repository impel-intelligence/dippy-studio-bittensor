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
- [Contributing](#contributing)
- [License](#license)

---

## Introduction

> **Note:** The following documentation assumes you are familiar with basic Bittensor concepts: Miners, Validators, and incentives. If you need a primer, please check out [Bittensor Docs](https://docs.bittensor.com/learn/bittensor-building-blocks)

Dippy is one of the world's leading AI companion apps with **7M+ users**. The app has ranked [**#3 on the App Store**](https://x.com/angad_ai/status/1850924240742031526) in countries like Germany, been covered by publications like [**Wired magazine**](https://www.wired.com/story/dippy-ai-girlfriend-boyfriend-reasoning/) and the average Dippy user **spends 1+ hour on the app.** 

When Dippy moved beyond text inference, we discovered that Bittensor still doesn’t have production-ready subnets for media inference. The Dippy Studio subnet aims to address this gap in the market and power the fastest, media generation platform on Bittensor. It's scalable, multi-modal, and ready to serve millions of users on Day 1. Our custom, in-house Dippy Inference Engine means that our users can reliably expect the fastest and cheapest inference available on the market.

## Roadmap 

Given the complexity of starting an enterprise grade inference engine from scratch, we plan to divide the process into 3 distinct phases.

**Phase 1:** 
- [ ] Subnet launch with Flux LoRA training and inference from Day 1
- [ ] Dippy Studio powers all Dippy in-chat images using character-specific LoRAs serving millions of users
- [ ] Beta version of self-serve API access for external builders to start building

**Phase 2:** 
- [ ] Release dashboard with live statistics on every model
- [ ] Set up our in-house Dippy Inference Engine pipeline to optimize any image model
- [ ] On-board 50+ AI startups with Dippy Studio's image inference

**Phase 3:** 
- [ ] Expand modalities to audio, video, 3D etc while maintaining our speed advantage
- [ ] Power 100+ AI startups with fully decentralized media infrastructure

## Overview of Miner and Validator Functionality

**Miners** would use existing frameworks, fine tuning techniques, or MergeKit, to train, fine tune, or merge models to create a unique roleplay LLM. These models would be submitted to a shared Hugging Face pool. 

**Validators** would evaluate the and assess model performance via our protocol and rank the submissions based on an [open scoring format](/docs/llm_scoring.md). We will provide a suite of 
testing and benchmarking protocols with state-of-the-art datasets.

## Running Miners and Validators

### Running a Miner
For detailed miner setup instructions, please see [Miner Documentation](docs/miner.md).

### Running a Validator
For detailed validator setup instructions, please see [Validator Documentation](docs/validator.md).

## Subnet Incentive Mechanism

The general structure of the incentive mechanism is as follows:
1. Every miner has a model registered per UID
2. Each miner's model submission is scored, with details outlined below
   - The scoring mechanism is constantly evolving according to SOTA model bechmark data

3. The validator compares each miner's score against all the other miners, and calculates a win rate
    - Note that there are some modifiers for a miner's score such as their submission age in relation to other miner submissions (aka time penalty) to combat blatant model copying
4. Given each miner's win rate, weights are assigned sorted by highest win rate


### Model Evaluation Criteria
See [scoring](/docs/llm_scoring.md) for details

## Subnet Token Management
See the [subnet token doc](/docs/subnet_token.md) for details

## Acknowledgement

Our codebase was originally built upon [Nous Research's](https://github.com/NousResearch/finetuning-subnet) and [MyShell's](https://github.com/myshell-ai/MyShell-TTS-Subnet?tab=readme-ov-file) Subnets. At the time of this writing, we have deviated significantly from these subnet architectures, providing more efficiency and capability.

## License

The Dippy Bittensor subnet is released under the [MIT License](./LICENSE).


# Project Structure Overview

## Core Components

### 1. Miner Implementation
- `dippy-studio-bittensor-miner/` - Main miner application

### 2. Subnet Infrastructure
- `subnet/` - Subnet-specific code and configuration

### 3. Documentation
- `docs/` - Project documentation
  - `miner.md` - Miner setup and usage guide
  - `validator.md` - Validator setup and usage guide
