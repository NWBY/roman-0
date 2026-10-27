import random


def generate_rnd(seed) -> random.Random:
    return random.Random(seed)


def generate_examples(rnd, count, min_x, max_x) -> list[tuple[float, float]]:
    examples = []
    for _ in range(count):
        x = rnd.uniform(min_x, max_x)
        y = (2.0 * x) + 1 + rnd.gauss(0, 0.3)
        examples.append((x, y))
    return examples


def shuffle_examples(rnd, examples) -> None:
    return rnd.shuffle(examples)


def split_examples(examples, split_index) -> list[tuple[float, float]]:
    return examples[:split_index], examples[split_index:]


def get_training_examples(examples) -> list[tuple[float, float]]:
    return examples[:70]
