"""
generate_dataset.py
-------------------
Generates a realistic synthetic Customer Support Ticket dataset and saves it to
data/customer_support_tickets.csv with the same column names the Kaggle dataset uses.

Run once:  python generate_dataset.py
"""

import os
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

# ── Column names matching Kaggle dataset & train.py constants ────────────────
TEXT_COL     = "Ticket Description"
CATEGORY_COL = "Ticket Type"
PRIORITY_COL = "Ticket Priority"

# ── Category definitions with realistic ticket templates ─────────────────────
TEMPLATES = {
    "Billing Issue": [
        "I was charged twice for my subscription this month. Please refund the duplicate charge.",
        "My invoice shows an incorrect amount. I should have been billed {amount} but was charged {amount2}.",
        "I cancelled my plan last week but was still billed. This is unacceptable.",
        "I need a copy of my invoice for {month} for my company records.",
        "The payment failed but money was still deducted from my account.",
        "I cannot update my credit card information on the billing page.",
        "There is an unauthorized charge of {amount} on my account. I did not authorize this.",
        "My coupon code was not applied to my bill. Please correct the amount.",
        "I upgraded my plan but was charged the full price without the promised discount.",
        "Please explain the breakdown of charges on my latest invoice.",
    ],
    "Technical Support": [
        "The application crashes every time I try to open the settings page.",
        "I am getting a 500 internal server error when submitting my form.",
        "The mobile app is extremely slow and keeps freezing on my phone.",
        "My dashboard is not loading any data. It just shows a spinning loader.",
        "I cannot upload files larger than 2MB even though my plan allows 50MB.",
        "The search feature returns no results even when I search for existing records.",
        "Two-factor authentication is not working. The code never arrives via SMS.",
        "The API integration with our CRM stopped working after the last update.",
        "My browser extension is not syncing data between devices.",
        "The export to PDF feature generates blank pages instead of my data.",
        "Video calls keep disconnecting after exactly 10 minutes.",
        "I cannot install the desktop application. The installer shows an error code 0x80.",
    ],
    "Account Management": [
        "I cannot log into my account. My credentials are correct but I keep getting an error.",
        "I need to transfer my account to a new email address.",
        "Please help me delete my account and all associated data permanently.",
        "I forgot my security question answer and cannot reset my password.",
        "My account was suspended without any notification. Please explain why.",
        "I need to add a secondary admin user to my organization account.",
        "How do I enable single sign-on (SSO) for my company's domain?",
        "I accidentally deleted my profile and need it restored from backup.",
        "I need to merge two accounts into one. Both use my email address.",
        "My account details page is not saving changes when I click update.",
    ],
    "Product Inquiry": [
        "Can you explain the difference between the Pro and Enterprise plans?",
        "Does your software support integration with Salesforce CRM?",
        "What is the maximum number of users allowed on the Business plan?",
        "Do you offer a free trial before I commit to a paid subscription?",
        "Is my data encrypted at rest and in transit on your platform?",
        "What file formats does the import feature support?",
        "Can I use your API without a paid plan?",
        "Does your tool support multi-language interfaces?",
        "What are the system requirements for installing your desktop application?",
        "Is there a limit on the number of projects I can create on the Basic plan?",
    ],
    "Shipping and Delivery": [
        "My order has been in processing for 5 days. When will it ship?",
        "The tracking number you provided shows no updates since last week.",
        "I received the wrong item in my shipment. Please send the correct one.",
        "My package was marked as delivered but I never received it.",
        "I need to change the delivery address before my order ships.",
        "The estimated delivery date has passed and my order still hasn't arrived.",
        "My package arrived damaged. I need a replacement sent immediately.",
        "Can I pick up my order from your warehouse instead of having it shipped?",
        "I ordered express shipping and was charged for it but my package arrived late.",
        "Please confirm whether you deliver to international addresses.",
    ],
    "Refund Request": [
        "I am not satisfied with the product and would like a full refund.",
        "I returned the item over two weeks ago and the refund hasn't appeared yet.",
        "My refund was only partial. I should receive the full {amount}.",
        "The product I received is defective. I need a refund or replacement.",
        "I cancelled within the 30-day guarantee but was told I don't qualify for a refund.",
        "I was charged for a service I never used. I want my money back.",
        "Please process my refund to the original payment method, not store credit.",
        "I upgraded accidentally and want to downgrade and get refunded the difference.",
        "The promotional offer was not applied and I want a refund for the overcharge.",
        "I have been waiting 15 business days for my refund. This is beyond your stated policy.",
    ],
    "Feedback and Suggestions": [
        "I love the product but the dashboard layout could be more intuitive.",
        "Please add dark mode support to the mobile application.",
        "It would be great if you added bulk export functionality.",
        "The onboarding tutorial is confusing for new users. Please simplify it.",
        "I suggest adding keyboard shortcuts to speed up navigation.",
        "Please integrate with Google Workspace. It would save us a lot of time.",
        "The notification emails are too frequent and there is no way to customize them.",
        "Your documentation needs more real-world code examples for the API.",
        "Please add a feature to schedule automatic reports on a weekly basis.",
        "The color contrast in the UI is too low, making it hard to read for users with visual impairments.",
    ],
}

PRIORITIES = ["High", "Medium", "Low"]
PRIORITY_WEIGHTS = [0.20, 0.50, 0.30]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
AMOUNTS = ["$29.99", "$49.99", "$99.00", "$149.00", "$199.99", "$9.99", "$14.99"]

CUSTOMER_NAMES = [
    "James Wilson", "Emma Johnson", "Liam Smith", "Olivia Brown", "Noah Davis",
    "Ava Miller", "William Moore", "Sophia Taylor", "Benjamin Anderson", "Isabella Thomas",
    "Elijah Jackson", "Mia White", "Lucas Harris", "Charlotte Martin", "Mason Thompson",
    "Amelia Garcia", "Ethan Martinez", "Harper Robinson", "Alexander Clark", "Evelyn Lewis",
]

STATUSES = ["Open", "In Progress", "Resolved", "Closed", "Pending"]
CHANNELS = ["Email", "Phone", "Chat", "Social Media", "Web Form"]
PRODUCTS = [
    "SupportPro v3", "BillingHub", "DataSync API", "MobileApp iOS",
    "MobileApp Android", "Desktop Client", "Web Dashboard", "Enterprise Suite",
]


def fill_template(template: str) -> str:
    """Fill placeholder values in a ticket template."""
    amount  = random.choice(AMOUNTS)
    amount2 = random.choice(AMOUNTS)
    while amount2 == amount:
        amount2 = random.choice(AMOUNTS)
    month = random.choice(MONTHS)
    return (template
            .replace("{amount}", amount)
            .replace("{amount2}", amount2)
            .replace("{month}", month))


def generate_dataset(n_tickets: int = 8500) -> pd.DataFrame:
    """Generate a synthetic support ticket DataFrame."""
    categories = list(TEMPLATES.keys())
    rows = []

    for i in range(1, n_tickets + 1):
        category  = random.choice(categories)
        template  = random.choice(TEMPLATES[category])
        text      = fill_template(template)
        priority  = np.random.choice(PRIORITIES, p=PRIORITY_WEIGHTS)
        customer  = random.choice(CUSTOMER_NAMES)
        status    = random.choice(STATUSES)
        channel   = random.choice(CHANNELS)
        product   = random.choice(PRODUCTS)

        rows.append({
            "Ticket ID":          f"TKT-{i:05d}",
            "Customer Name":      customer,
            TEXT_COL:             text,
            CATEGORY_COL:         category,
            PRIORITY_COL:         priority,
            "Ticket Status":      status,
            "Channel":            channel,
            "Product Purchased":  product,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("[generate] Creating synthetic dataset ...")
    df = generate_dataset(n_tickets=8500)
    out_path = os.path.join("data", "customer_support_tickets.csv")
    df.to_csv(out_path, index=False)
    print(f"[generate] Saved {len(df):,} tickets to: {out_path}")
    print(f"[generate] Category distribution:\n{df['Ticket Type'].value_counts().to_string()}")
    print(f"\n[generate] Priority distribution:\n{df['Ticket Priority'].value_counts().to_string()}")
