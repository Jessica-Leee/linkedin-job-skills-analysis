# LinkedIn Job Analysis

An exploratory Python project for studying skills mentioned in semiconductor and design-engineering job postings. It combines browser collection helpers with CSV cleaning, skill extraction, topic exploration, and report generation.

## What is implemented

- Job search and pagination, job details, company information, and profile collection helpers.
- Missing-value handling, title normalization, and posting-month extraction.
- Technical-skill dictionaries and comparisons by role, location, and month.
- TF-IDF, LDA, NMF, and KMeans exploration; skill charts, word clouds, and co-occurrence heatmaps.

The analysis uses standard methods from pandas, NLTK, scikit-learn, matplotlib, seaborn, and WordCloud. The bundled Selenium collection components are not presented as a new scraping framework. Preserve applicable upstream notices when reusing them.

## Setup

Use Python 3.10 or 3.11 for the original pinned data-science dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Try the synthetic example

[`examples/synthetic_jobs.csv`](examples/synthetic_jobs.csv) contains 60 entirely invented postings. Its companies, locations, descriptions, and skill patterns are fixtures, not job-market observations.

Clean the sample without NLP reports or NLTK downloads:

```sh
python src/analysis/pre-process.py --clean-only
```

Run the full exploratory report after installing dependencies:

```sh
python src/analysis/pre-process.py --input examples/synthetic_jobs.csv --output-dir outputs/demo
```

The command writes `cleaned_jobs.csv` and, for the full workflow, a `skills_report/` directory beneath the selected output directory. Full analysis may download NLTK stopword and WordNet resources on first use.

For your own CSV, provide `job_title`, `company`, `location`, `posted_date`, and `job_description` columns and select it with `--input`.

## Live collection

Set `LINKEDIN_EMAIL` and `LINKEDIN_PASSWORD` in the process environment. `.env.example` documents the names; the collector does not automatically read that template or a `.env` file. Do not put configuration values in source control.

With Chrome and the dependencies installed:

```sh
python -m src.scrapers.main --query "semiconductor engineer" --max-pages 3 --output outputs/jobs.csv
```

`--help` and module imports do not launch Chrome. Missing configuration stops the command before browser dependencies are loaded. The browser is closed when collection finishes or raises an error.

## Offline checks

No third-party packages, browser, account, or network access are needed for these checks:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover configuration validation, browser-free imports/help, missing-authentication handling, synthetic input paths, and CSV round trips. They do not run Selenium or evaluate the NLP models.

## Limitations

The full dependency installation, live collection, and end-to-end analytical report have not been validated by the offline suite. LinkedIn page changes or sign-in challenges can require collector updates. Generated topics and counts describe the supplied corpus; the synthetic example and unsupervised methods do not establish labor-market trends or predictive accuracy.
