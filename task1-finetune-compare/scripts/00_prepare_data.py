"""Create the train/val/test splits used by BOTH models.

Usage:
    python scripts/00_prepare_data.py                      # synthetic demo data
    python scripts/00_prepare_data.py --csv my_tickets.csv # your Part 1 data
                                                           # (columns: text,category,priority)
"""
import argparse
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from common import DATA, SEED, SPLITS

TEMPLATES = {
    "Billing": [
        ("I was charged twice for my subscription", "High"),
        ("The payment failed but money was deducted from my card", "High"),
        ("My refund has not arrived yet", "Medium"),
        ("Why did the price of my plan increase", "Medium"),
        ("Please send me an invoice for last month", "Low"),
        ("Update my billing address on the invoice", "Low"),
    ],
    "Technical": [
        ("The app crashes every time I open it", "High"),
        ("The whole website is down and we cannot work", "High"),
        ("I get an error when uploading a file", "Medium"),
        ("The dashboard loads very slowly", "Medium"),
        ("The export button does nothing", "Medium"),
        ("How do I enable dark mode in settings", "Low"),
    ],
    "Account": [
        ("I cannot log in to my account", "High"),
        ("My account was hacked and someone changed my password", "High"),
        ("I did not receive the verification email", "Medium"),
        ("I want to delete my account", "Medium"),
        ("Please change the email address on my profile", "Low"),
        ("How do I add a new team member", "Low"),
    ],
    "Shipping": [
        ("My order has not arrived and it was due last week", "High"),
        ("The package came damaged", "High"),
        ("Can I change the delivery address", "Medium"),
        ("Tracking shows no updates for five days", "Medium"),
        ("What is the status of my shipment", "Low"),
        ("Do you ship to Canada", "Low"),
    ],
    "General": [
        ("Can I speak to a manager about my experience", "Medium"),
        ("What are your support hours", "Low"),
        ("Do you offer student discounts", "Low"),
        ("I would like to give feedback on your service", "Low"),
        ("Where can I find your terms of service", "Low"),
        ("Do you have a partnership program", "Low"),
    ],
}
PREFIX = ["", "Hi,", "Hello team,", "Good morning,", "Hey,"]
SUFFIX = ["", "Thanks.", "Please help.", "Any update?", "Thank you!", "Regards."]
EXTRA = ["", "", "It happened again today.", "This is the second time.",
         "I use the mobile app.", "I am on the premium plan."]


def synthetic(n=3000, seed=SEED):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        cat = rng.choice(list(TEMPLATES))
        base, prio = rng.choice(TEMPLATES[cat])
        text = " ".join(
            p for p in (rng.choice(PREFIX), base + ".", rng.choice(EXTRA),
                        f"Ref #{rng.randint(1000, 9999)}." if rng.random() < .3 else "",
                        rng.choice(SUFFIX)) if p
        )
        if rng.random() < 0.08:  # urgency cue bumps priority
            text = "URGENT: " + text
            prio = "High"
        if rng.random() < 0.2:
            text = text.lower()
        rows.append({"text": text, "category": cat, "priority": prio})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="CSV with columns text,category,priority")
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else synthetic()
    df = df[["text", "category", "priority"]].dropna()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    train, rest = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["category"])
    val, test = train_test_split(rest, test_size=0.50, random_state=SEED, stratify=rest["category"])

    SPLITS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(exist_ok=True)
    df.to_csv(DATA / "tickets.csv", index=False)
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.to_csv(SPLITS / f"{name}.csv", index=False)
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    print(df["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
