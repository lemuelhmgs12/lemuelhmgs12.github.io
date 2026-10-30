---
layout: post
title: Building an Incremental Data Pipeline with S3, DuckDB and Airflow
image: "/posts/openfoodfacts.png"
tags: [Python, Airflow, DuckDB, AWS, Terraform, Data Engineering]
---

I built a small, production-style pipeline that keeps a product dataset current without reprocessing everything on every run. It pulls snack products from the Open Food Facts API, lands the raw data in S3, deduplicates it with DuckDB, and builds a daily rollup. Code: [openfoodfactsd-data-pipeline](https://github.com/lemuelhmgs12/openfoodfacts-data-pipeline)

---

## The problem

Product-catalog teams need to see what changed (new items, updated ingredients, category shifts) without reloading the whole catalog each time. The goal was a pipeline that fetches only what changed since the last run, tolerates a flaky source, and can be re-run safely.

## Architecture

![Pipeline architecture]({{ '/img/posts/pipeline_architecture.png' | relative_url }})

An Airflow DAG runs daily with three tasks:

1. **ingest** pulls changed products from the API and writes raw JSON to S3.
2. **dedupe** (DuckDB) reads the raw files straight from S3 and writes one row per product to a staged Parquet file.
3. **rollup** (DuckDB) counts products modified per UTC day and writes a marts Parquet file.

Airflow controls the order; S3 holds the data. Each task runs as its own process, so tasks never share memory. They hand off through S3, and Airflow passes only small values such as a file path.

I chose DuckDB over Glue/Spark with Redshift or Athena because a few thousand rows don't justify a cluster. DuckDB reads JSON and Parquet from S3 with plain SQL files and has nothing to run.

## Making it incremental

- A **watermark** (the newest `last_modified_t` seen) is stored as JSON in S3.
- Results are requested newest-first, which I verified by checking that timestamps come back in descending order. Paging stops as soon as it reaches a product at or older than the watermark.
- The comparison is `>=`, so products exactly at the watermark are fetched again. The raw zone is therefore at-least-once, and the dedupe step removes duplicates downstream.
- The watermark moves forward **only after every batch is written**. If a run crashes halfway, the watermark is untouched and the run can simply be repeated. I confirmed this directly in S3, not just in tests.

## What went wrong, and what I did about it

**The API fails a lot.** I measured about a 60% rate of 503 responses. At that rate retries are essential, not optional, so requests use exponential backoff (up to 5 attempts). If retries run out, the run fails loudly and the watermark stays put. Silently skipping would have lost data on almost every run.

**The real page size is 100.** I had asked for 1,000 products per page, but the API silently caps it at 100. That made my batches rarely complete, because five successful pages in a row were needed against a high failure rate. I matched the page size to the real limit and lowered the batch size, so a batch now completes in three pages.

**Page 11 returns a 401.** Runs under Airflow kept failing after page 10, and direct requests showed the limit is on page number, not result count. At 100 per page, at most 1,000 products are reachable per query. I capped each run at 10 pages and added a warning when the cap ends a run before the watermark is reached, so a gap is visible instead of silent.

**Same-day runs overwrote each other.** Batch files were named by date and a counter, so a second run on the same day replaced the first run's files. I only noticed by looking at S3 and seeing files being replaced instead of piling up. Adding the run time to the filename fixed it.

**A time zone changed the numbers.** The same data gave different daily counts depending on whether dates were cut in local time or UTC, because the conversion uses the session's time zone (Los Angeles on my laptop, UTC in the container). The rollup now converts to UTC explicitly, so results don't depend on where the query runs.

## Safe to re-run

The staged and marts files are written to fixed keys and rebuilt in full from the raw data, so an Airflow retry produces the same file. As a check, the staged row count matched the number of distinct product codes in raw.

## Tests and CI

The pipeline has pytest tests (API client, watermark, orchestration and SQL) at 91% coverage. GitHub Actions runs them on every push and pull request, and fails the build if coverage drops below 90%.

## Access control with Terraform

The pipeline doesn't run with admin keys. A Terraform-managed IAM role allows only what the job needs:

- `s3:ListBucket` on the bucket
- `s3:GetObject` and `s3:PutObject` on the objects in it

These are separate statements because they apply to different resource types (the bucket and the objects inside it). The pipeline assumes the role through AWS Security Token Service (STS) and gets temporary credentials, so a leaked credential could only touch that one bucket and would expire on its own. I confirmed the role can read the bucket and is denied other actions such as listing IAM users. Locally the role is reached through an AWS config profile; on a deployed host I would attach it as an instance profile instead.

## What I would do next

The documentation steers heavy use away from the search API. Open Food Facts publishes daily delta files, and a day's file held roughly 5,600 products, about 10% of them snacks. Moving to those would remove the 1,000-product ceiling and let a pipeline that was down for several days catch up by reading the missed files in order, as long as it returns within the 14-day retention. The decision and the design are written up in the repo; I have deferred the build.
