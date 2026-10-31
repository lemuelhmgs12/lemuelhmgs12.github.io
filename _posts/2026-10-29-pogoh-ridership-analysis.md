---
layout: post
title: POGOH Ridership Analysis - E-bike Adoption and the 2025 Pricing Change
image: "/posts/pogoh_ridership_cover.png"
tags: [Python, Jupyter, Data Analysis, Pittsburgh]
---

POGOH is Pittsburgh's bike-share system. I analyzed its historical trip data to understand how e-bike adoption and ridership have grown, and what happened after the 2025 pricing restructure. The analysis behind the strategic roadmap is in the repo: https://github.com/lemuelhmgs12/pogoh-ridership-analysis

---

## The questions

- How has ridership grown over time?
- How quickly did riders adopt e-bikes?
- What changed after the 2025 pricing restructure?

## The data

984,180 trips from the WPRDC POGOH data, May 2022 to July 2025 (2025 is only a partial year), plus three station files. The work is a set of numbered notebooks, and uv sets up the environment..

## How the analysis is organized

The work is a series of numbered Jupyter notebooks, run in order. Notebook 02 builds the processed files from the raw data, and notebooks 03 to 07 run the analysis on them. Figures and statistical outputs are saved to `results/`. Python dependencies are managed with uv, so the environment can be recreated from the repo.

## Cleaning Step

Cleaning took the data from 984,180 to 935,004 trips, about 5%:

- Removed 48,489 false starts: rides under 60 seconds that began and ended at the same station, plus 107 zero-second terminated rides.
- Set aside, not deleted: 494 terminated rides with no end station and 188 rides over 24 hours. They sit in their own dataframes in case they are needed later.
- Dropped Rider Id, which only exists for 2023, and used account ID instead.
- Removed 4 pre-launch test records and 1 unmatched station record.

## What I found

![Actual monthly rides vs the pre-2025 trend, with the gap after the Flex Pass retirement](/img/posts/pogoh_growth_gap.png)
- **E-assist bikes are 81% of rides** (95% CI 81.30% to 81.45%). Members ride them 83.7% of the time, casual riders 62.8%. That is a 20.9-point gap, and it shows association, not cause.
- **Growth slowed sharply.** Comparing January to July each year, rides grew 200.9% from 2023 to 2024 but only 17.1% from 2024 to 2025. Flex Pass rides fell from 401,800 in 2024 to 2,623 in 2025 (a full year versus seven months).
- **A simple before/after average misleads.** It shows rides up 48.6%, which is just long-term growth. Against a trend fitted on pre-2025 data, January 2025 came in 59.2% below prediction and recovered to 31% below by July.
- **Three stations to expand:** S Bouquet Ave & Sennott St (3,645 rides per dock, 4.5 times the system average), Atwood St & Bates St (2,718), and Allequippa St & Darragh St (1,374, which extends coverage west). One early weighting suggested Wilkinsburg Park & Ride at only 41 rides per dock, so I added a 500 rides-per-dock minimum.



## Method notes

- A linear model with a seasonal term fit the pre-2025 months best (train R-squared 0.75). It missed my 0.80 target on the test months, and polynomial models did worse (test R-squared from -1.66 to -10.99). The January 2025 pricing change broke the pattern, so nothing trained on earlier months predicts 2025 well.
- An interrupted time series put the drop at about 12,560 rides a month, but with only seven months of 2025 it is not statistically significant (p = 0.119). I treat it as directional.
- E-bike comparisons used chi-square tests and a 1,000-resample bootstrap.

## What I would do next

- Rerun the interrupted time series once a full year of 2025 data exists.
- Fit separate models for before and after the change.
- Review the three stations with POGOH's operations team before anyone acts on them.

## Reproduce it

```bash
uv sync
# run notebook 02 first, then 03 to 07 in order
```
