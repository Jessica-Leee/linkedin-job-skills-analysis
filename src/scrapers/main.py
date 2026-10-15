"""Collect job listings only when explicitly invoked as a command."""

import argparse
import csv
import math
import os
from pathlib import Path
from time import sleep


def save_dict_list_to_csv(data, filename, encoding='utf-8'):
    """
    Save a list of dictionaries to a CSV file.
    
    Args:
        data (list): List of dictionaries where each dictionary represents a row
        filename (str): Path to the output CSV file
        encoding (str, optional): File encoding. Defaults to 'utf-8'.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not data or not isinstance(data, list):
        print("Error: Data must be a non-empty list of dictionaries")
        return False
    
    try:
        # Get fieldnames from the keys of the first dictionary
        fieldnames = data[0].keys()
        
        with open(filename, 'w', newline='', encoding=encoding) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header row
            writer.writeheader()
            
            # Write the data rows
            writer.writerows(data)
        
        print(f"Successfully saved {len(data)} rows to {filename}")
        return True
    
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Job-search query")
    parser.add_argument("--max-pages", type=positive_int, default=30)
    parser.add_argument("--delay", type=nonnegative_float, default=1.0,
                        help="Seconds between job-detail requests")
    parser.add_argument("--output", type=Path, default=Path("outputs/jobs.csv"))
    parser.add_argument("--headless", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.query.strip():
        parser.error("--query must not be blank")
    email = os.environ.get("LINKEDIN_EMAIL", "").strip()
    password = os.environ.get("LINKEDIN_PASSWORD", "")
    if not email or not password:
        parser.error("Set LINKEDIN_EMAIL and LINKEDIN_PASSWORD in the process environment")

    # Import browser dependencies after configuration validation. Importing this
    # module or requesting --help never starts Chrome or downloads a driver.
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from .job_search import JobSearch
    from . import actions

    options = webdriver.ChromeOptions()
    if args.headless:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        actions.login(driver, email, password)
        search = JobSearch(driver=driver, close_on_complete=False, scrape=False)
        listings = search.search_all_pages(args.query, max_pages=args.max_pages)
        results = []
        for job in listings:
            try:
                job.scrape()
                results.append(job.to_dict())
            except Exception as exc:
                print(f"Could not collect job details: {exc}")
            sleep(args.delay)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        return 0 if save_dict_list_to_csv(results, args.output) else 1
    finally:
        driver.quit()


if __name__ == "__main__":
    raise SystemExit(main())
