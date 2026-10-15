# LinkedIn Job Analysis

An exploratory Python project for examining skills mentioned in semiconductor and design-engineering job postings. It combines collection utilities with text cleaning, skill extraction, topic exploration, and report generation.

## Implemented

- Selenium helpers for job searches, pagination, job details, companies, and profiles.
- CSV cleaning, title normalization, missing-value handling, and posting-month extraction.
- Technical-skill dictionary matching and comparisons by role, location, and time.
- TF-IDF features, LDA and NMF topic exploration, and KMeans clustering.
- Skill-frequency charts, word clouds, co-occurrence heatmaps, and generated text reports.

The main analytical workflow is in `src/analysis/pre-process.py`. The collection components are under `src/scrapers/`.

## Setup

Create an environment and install the listed dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency file is incomplete. Direct imports also require `nltk`, `scikit-learn`, `wordcloud`, `webdriver-manager`, and `lxml`. Reconcile these dependencies before running the scripts. The analysis setup may download NLTK resources on first use.

For analysis, supply a job-posting CSV and update the paths in `main()`. The current defaults expect `marvell_jobs_final.csv` in the working directory and write `marvell_jobs_final_cleaned.csv`. The cleaner expects fields including `job_title`, `company`, `location`, and `posted_date`; the report also uses job-description text.

After preparing the input and environment:

```sh
python src/analysis/pre-process.py
```

Generated reports and charts are written to `emerging_skills_report/`.

## Current limitations

The repository does not include a sample CSV or a checked-in results package. Its charts and topic models therefore cannot be reproduced from a clone alone, and no validated job-market findings or predictive scores are reported here.

The scraping entry point imports `linkedin_scraper`, while the bundled directory is named `src/scrapers`; its package layout and imports need reconciliation before the documented collection command can run. Authentication must be supplied through local configuration rather than copied into source or examples.

The scraper is browser- and page-structure dependent. The `tests/` directory contains a scraping example, not a verified offline regression suite. Preserve applicable upstream notices when reusing the collection components.
