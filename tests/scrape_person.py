"""Opt-in live collection example; this is not an offline test."""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_url")
    args = parser.parse_args()
    from src.scrapers import Person
    print(Person(args.profile_url))


if __name__ == "__main__":
    main()
