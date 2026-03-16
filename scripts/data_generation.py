"""
Generate a synthetic Hinglish delivery dataset.

Produces hinglish_delivery_dataset_1000.json in data/ (or current dir if
data/ is not found from the working directory).

Usage (from project root)
--------------------------
  python -m scripts.data_generation
"""

import json
import random
from pathlib import Path

TOTAL_SAMPLES = 1000

INTENTS = [
    "get_address",
    "call_customer",
    "mark_delivered",
    "mark_picked_up",
    "report_delay",
    "navigation_help",
    "order_issue",
    "customer_unavailable",
]

order_refs    = ["next", "agla", "current", "agle", "next wala"]
delay_times   = [5, 7, 10, 12, 15, 20]
customer_words = ["customer", "custmr", "cstmr"]
address_words  = ["address", "adress", "addres"]

TEMPLATES: dict[str, list[str]] = {
    "get_address": [
        "Bhai {order} order ka {address} bhej do",
        "{order} order ka location kya hai",
        "pls {order} order ka adress send karo",
        "zara {order} delivery ka address dikhao",
        "bhai {order} order ka exact address batao",
        "yaar {order} order ka location chahiye",
        "mujhe {order} order ka address de do",
        "{order} order ka address kidhar hai",
        "bhai {order} delivery ka addres bata",
        "{order} order ka location bhejna",
    ],
    "call_customer": [
        "{customer} ko call karo",
        "bhai {customer} ko phone laga do",
        "{customer} ko ek baar ring karo",
        "pls {customer} ko call connect karo",
        "{customer} ko phone karna hai",
        "bhai zara {customer} ko call lagao",
        "{customer} ko abhi phone karo",
        "{customer} ko ek baar contact karo",
        "{customer} ko jaldi call karo",
        "{customer} ko ring lagao",
    ],
    "mark_delivered": [
        "order deliver ho gaya mark karo",
        "delivery complete ho gayi update kar do",
        "customer ko order de diya delivered mark karo",
        "bhai delivery done mark kar do",
        "order deliverd hogya update kro",
        "parcel customer ko de diya",
        "delivery complete ho gayi",
        "order successfully deliver kar diya",
        "customer ko order de diya update karo",
        "delivery ho gayi mark delivered",
    ],
    "mark_picked_up": [
        "order pickup kar liya restaurant se",
        "food pick kar liya mark pickup",
        "restaurant se order le liya update karo",
        "pickup done from hotel",
        "order pickd up hogya",
        "parcel restaurant se le liya",
        "restaurant se food collect kar liya",
        "pickup ho gaya update kar do",
        "order le liya restaurant se",
        "bhai order pickup kar liya",
    ],
    "report_delay": [
        "{time} min late ho jaunga",
        "traffic hai {time} minute aur lagenge",
        "customer ko bol do {time} min delay hoga",
        "road block hai {time} min late aaunga",
        "{time} minute delay ho raha hai",
        "thoda late hu {time} min aur",
        "jam hai yaha {time} min late aaunga",
        "{time} minute aur lagenge pahunchne mein",
        "bhai {time} min delay ho gaya",
        "traffic heavy hai {time} min aur",
    ],
    "navigation_help": [
        "map navigation start karo",
        "route dikhao customer location ka",
        "map kholo aur rasta batao",
        "navigation on karo pls",
        "customer address ka direction chahiye",
        "best route dikhao delivery ke liye",
        "location ka rasta bata do",
        "map pe route open karo",
        "navigation help chahiye location ke liye",
        "customer tak ka rasta dikhao",
    ],
    "order_issue": [
        "restaurant bol raha order ready nahi hai",
        "order me item missing lag raha hai",
        "order galat pack hua lagta hai",
        "packet damage lag raha hai",
        "restaurant ne bola order ready nahi",
        "order me problem hai check karo",
        "item missing lag raha packet me",
        "order ka packet thoda damage hai",
        "restaurant delay kar raha order",
        "order sahi pack nahi hua lagta",
    ],
    "customer_unavailable": [
        "customer phone nahi utha raha",
        "location pe hu customer nahi mil raha",
        "customer reachable nahi hai",
        "custmr call pick nahi kar raha",
        "ghar pe koi nahi mil raha",
        "customer address pe nahi hai",
        "customer ka phone band hai",
        "customer mil nahi raha location pe",
        "door pe hu par customer nahi hai",
        "customer respond nahi kar raha",
    ],
}

SLOT_DEFAULTS: dict[str, dict] = {
    "get_address":          {"order_reference": None},   # sampled from order_refs
    "call_customer":        {"target": "customer"},
    "mark_delivered":       {"status": "delivered"},
    "mark_picked_up":       {"status": "picked_up"},
    "report_delay":         {"delay_time": None, "unit": "minutes"},  # sampled
    "navigation_help":      {"navigation_action": "show_route"},
    "order_issue":          {"issue_type": "order_problem"},
    "customer_unavailable": {"availability": "unreachable"},
}


def generate(n: int = TOTAL_SAMPLES, seed: int = 42) -> list[dict]:
    random.seed(seed)
    samples_per_intent = n // len(INTENTS)
    dataset: list[dict] = []

    for intent in INTENTS:
        for _ in range(samples_per_intent):
            template = random.choice(TEMPLATES[intent])
            text     = template.format(
                order=random.choice(order_refs),
                address=random.choice(address_words),
                customer=random.choice(customer_words),
                time=random.choice(delay_times),
            )
            slots = dict(SLOT_DEFAULTS[intent])
            if intent == "get_address":
                slots["order_reference"] = random.choice(order_refs)
            if intent == "report_delay":
                slots["delay_time"] = random.choice(delay_times)
            dataset.append({"text": text, "intent": intent, "slots": slots})

    random.shuffle(dataset)
    return dataset


def main() -> None:
    data    = generate()
    out_dir = Path("data") if Path("data").exists() else Path(".")
    out     = out_dir / "hinglish_delivery_dataset_1000.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(data)} samples → {out}")


if __name__ == "__main__":
    main()
