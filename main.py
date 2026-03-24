#!/usr/bin/env python3
"""
Minimal Academic Note Summarizer CLI application.

This script provides:
1) Extractive summarization
2) Simple quiz generation from key sentences
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of",
    "in", "on", "at", "by", "with", "as", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "from", "into", "about", "than",
    "can", "could", "should", "would", "will", "may", "might", "do", "does", "did",
    "not", "no", "yes", "we", "you", "they", "he", "she", "them", "our", "your",
}


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> Iterable[str]:
    return re.findall(r"[A-Za-z]{3,}", text.lower())


def summarize(text: str, sentence_count: int = 5) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= sentence_count:
        return text.strip()

    word_counts = Counter(token for token in tokenize(text) if token not in STOPWORDS)
    if not word_counts:
        return " ".join(sentences[:sentence_count])

    scored = []
    for index, sentence in enumerate(sentences):
        words = [w for w in tokenize(sentence) if w not in STOPWORDS]
        score = sum(word_counts[w] for w in words) / max(len(words), 1)
        scored.append((index, score))

    selected_indices = sorted(index for index, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:sentence_count])
    return " ".join(sentences[i] for i in selected_indices)


def generate_quiz(text: str, question_count: int = 5) -> list[dict[str, str | int]]:
    sentences = split_sentences(text)
    ranked_sentences = sorted(
        sentences,
        key=lambda s: len([w for w in tokenize(s) if w not in STOPWORDS]),
        reverse=True,
    )
    chosen = ranked_sentences[:question_count]

    quiz = []
    for idx, sentence in enumerate(chosen, start=1):
        original_words = re.findall(r"[A-Za-z]{3,}", sentence)
        words = [w for w in original_words if w.lower() not in STOPWORDS]
        if not words:
            continue
        keyword = max(words, key=lambda w: len(w))
        pattern = re.compile(rf"\b{re.escape(keyword)}\b", flags=re.IGNORECASE)
        match = pattern.search(sentence)
        if not match:
            continue
        answer = match.group(0)
        masked_sentence = pattern.sub("_____", sentence, count=1)
        quiz.append(
            {
                "id": idx,
                "question": f"Fill in the blank: {masked_sentence}",
                "answer": answer,
            }
        )
    return quiz


def read_input(text: str | None, file_path: str | None) -> str:
    if text:
        return text.strip()
    if file_path:
        return Path(file_path).read_text(encoding="utf-8").strip()
    raise ValueError("Provide either --text or --file.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Academic Note Summarizer CLI")
    parser.add_argument("--text", type=str, help="Input text")
    parser.add_argument("--file", type=str, help="Path to input .txt file")
    parser.add_argument("--summary-sentences", type=int, default=5, help="Number of summary sentences")
    parser.add_argument("--quiz-questions", type=int, default=5, help="Number of quiz questions")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        source_text = read_input(args.text, args.file)
    except (ValueError, FileNotFoundError, UnicodeDecodeError, OSError) as exc:
        print(f"Error: {exc}")
        return 1

    if not source_text:
        print("Error: Input text is empty.")
        return 1

    summary = summarize(source_text, max(args.summary_sentences, 1))
    quiz = generate_quiz(source_text, max(args.quiz_questions, 1))

    print("\n=== SUMMARY ===")
    print(summary)

    print("\n=== QUIZ ===")
    if not quiz:
        print("Not enough content to generate quiz questions.")
    else:
        for item in quiz:
            print(f"\nQ{item['id']}: {item['question']}")
            print(f"Answer: {item['answer']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
