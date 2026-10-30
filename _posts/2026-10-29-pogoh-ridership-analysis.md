---
layout: post
title: POGOH Ridership Analysis - E-bike Adoption and the 2025 Pricing Change
image: "/posts/pogoh_ridership.png"
tags: [Python, Jupyter, Data Analysis, Pittsburgh]
---

POGOH is Pittsburgh's bike-share system. I analyzed its historical trip data to understand how e-bike adoption and ridership have grown, and what happened after the 2025 pricing restructure. The analysis behind the strategic roadmap is in the repo: https://github.com/lemuelhmgs12/pogoh-ridership-analysis

---

## The questions

- How has ridership grown over time?
- How quickly did riders adopt e-bikes?
- What changed after the 2025 pricing restructure?

## The data

POGOH Trip Data from the Western Pennsylvania Regional Data Center (WPRDC), covering May 2022 through 2025. The raw CSV files are in the repo under `data/raw/`.

## How the analysis is organized

The work is a series of numbered Jupyter notebooks, run in order. Notebook 02 builds the processed files from the raw data, and notebooks 03 to 07 run the analysis on them. Figures and statistical outputs are saved to `results/`. Python dependencies are managed with uv, so the environment can be recreated from the repo.

[ADD: one or two sentences on your cleaning steps, for example how you handled missing values, duplicate trips or station changes.]

## What I found

[ADD: your two or three headline findings, each with one number. For example, how the e-bike share of trips changed, how ridership growth looked year over year, and the difference in ridership before and after the pricing change.]

[ADD: one chart, saved to `/posts/pogoh_ridership.png`, with a one-line caption.]

## Method notes

[ADD: how you measured the pricing effect (a before/after comparison, a seasonal adjustment, a statistical test) and the main caveat, such as seasonality or other changes in 2025 that could also explain the difference.]

## What I would do next

[ADD: one or two follow-ups, such as station-level analysis or a forecast for the next season.]

## Reproduce it

```bash
uv sync
# run notebook 02 first, then 03 to 07 in order
```
