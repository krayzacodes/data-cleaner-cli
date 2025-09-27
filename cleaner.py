#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Cleaner CLI
Author: Mizgin Y.
Repo: https://github.com/krayzacodes/data-cleaner-cli

A powerful yet lightweight command-line tool to clean CSV files:
- drop duplicates
- trim spaces
- handle missing values (numeric & text)
- coerce numeric & date columns
- normalize emails
- select / drop / rename columns
- dataset summary
- change report (Markdown)
"""

from __future__ import annotations
import argparse
import sys
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd


# --------- helpers -----------------------------------------------------------

def _exists_or_die(path: Path) -> None:
    if not path.exists():
        sys.exit(f"Error: file not found: {path!s}")


def _parse_csv_list(s: Optional[str]) -> List[str]:
    """Comma or semicolon separated -> list (handles spaces)."""
    if not s:
        return []
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]


def _parse_mapping(pairs: Optional[List[str]]) -> Dict[str, str]:
    """
    Parse --rename pairs like:
      --rename old1=new1 --rename old2=new2
    """
    mapping: Dict[str, str] = {}
    if not pairs:
        return mapping
    for raw in pairs:
        if "=" not in raw:
            sys.exit(f"--rename must be in old=new form. Got: {raw!r}")
        old, new = raw.split("=", 1)
        mapping[old.strip()] = new.strip()
    return mapping


def _coerce_numeric(df: pd.DataFrame, cols: List[str], errors: str = "coerce") -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Convert given columns to numeric. Returns df and counts of coerced values.
    """
    counts: Dict[str, int] = {}
    for c in cols:
        if c not in df.columns:
            print(f"[warn] numeric coerce: column not found: {c}")
            continue
        before_na = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors=errors)
        after_na = df[c].isna().sum()
        counts[c] = max(0, after_na - before_na)
    return df, counts


def _coerce_date(df: pd.DataFrame, cols: List[str], dayfirst: bool) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Convert given columns to datetime (infer). Returns df and invalid counts.
    """
    counts: Dict[str, int] = {}
    for c in cols:
        if c not in df.columns:
            print(f"[warn] date coerce: column not found: {c}")
            continue
        before_na = df[c].isna().sum()
        df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True, dayfirst=dayfirst)
        after_na = df[c].isna().sum()
        counts[c] = max(0, after_na - before_na)
    return df, counts


def _normalize_emails(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Normalize email-like values:
      - replace '[at]' with '@'
      - lowercase
      - strip spaces
    Returns df and counts of modified cells.
    """
    email_fix_re = re.compile(r"\s*\[\s*at\s*\]\s*", flags=re.I)

    changed: Dict[str, int] = {}
    for c in cols:
        if c not in df.columns:
            print(f"[warn] email normalize: column not found: {c}")
            continue
        cnt = 0

        def _fix(v):
            nonlocal cnt
            if isinstance(v, str):
                new = email_fix_re.sub("@", v).strip().lower()
                if new != v:
                    cnt += 1
                return new
            return v

        df[c] = df[c].map(_fix)
        changed[c] = cnt
    return df, changed


def _trim_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing spaces in all string cells."""
    return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


def _summary(df: pd.DataFrame) -> str:
    parts: List[str] = []
    parts.append(f"Rows: {len(df):,}")
    parts.append(f"Columns: {len(df.columns)}")
    parts.append("Missing values per column:")
    miss = df.isna().sum().sort_values(ascending=False)
    if not miss.empty:
        for k, v in miss.items():
            parts.append(f"  - {k}: {int(v)}")
    else:
        parts.append("  (none)")
    return "\n".join(parts)


# --------- core cleaning pipeline -------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    *,
    trim: bool,
    drop_blank_rows: bool,
    dedupe: bool,
    fill_text: Optional[str],
    fill_numeric: Optional[str],
    fill_numeric_value: Optional[float],
    coerce_numeric_cols: List[str],
    coerce_date_cols: List[str],
    dayfirst: bool,
    email_cols: List[str],
    select_cols: List[str],
    drop_cols: List[str],
    rename_map: Dict[str, str],
    verbose: bool,
    change_log: List[str],
) -> pd.DataFrame:
    """Apply all requested transformations to the dataframe."""
    if trim:
        df = _trim_strings(df)
        change_log.append("Trimmed whitespace in all string columns")

    # Replace empty strings with NaN (standardize missing)
    empty_to_nan = (df == "").sum().sum()
    if empty_to_nan:
        df.replace("", np.nan, inplace=True)
        change_log.append(f"Replaced {int(empty_to_nan)} empty strings with NaN")

    if drop_blank_rows:
        before = len(df)
        df.dropna(how="all", inplace=True)
        removed = before - len(df)
        change_log.append(f"Dropped {removed} completely blank rows")

    if dedupe:
        before = len(df)
        df.drop_duplicates(inplace=True)
        removed = before - len(df)
        change_log.append(f"Removed {removed} duplicate rows")

    if coerce_numeric_cols:
        df, counts = _coerce_numeric(df, coerce_numeric_cols, errors="coerce")
        total = sum(counts.values())
        change_log.append(f"Coerced numeric columns {coerce_numeric_cols} (new NaNs created: {total})")

    if coerce_date_cols:
        df, counts = _coerce_date(df, coerce_date_cols, dayfirst=dayfirst)
        total = sum(counts.values())
        change_log.append(f"Coerced date columns {coerce_date_cols} (invalids -> NaT: {total})")

    if email_cols:
        df, changed = _normalize_emails(df, email_cols)
        total = sum(changed.values())
        change_log.append(f"Normalized emails in {email_cols} (cells changed: {total})")

    # Fill missing values
    if fill_numeric:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if fill_numeric == "mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean(numeric_only=True))
            change_log.append(f"Filled numeric NaNs with column MEANs: {num_cols}")
        elif fill_numeric == "median":
            df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))
            change_log.append(f"Filled numeric NaNs with column MEDIANs: {num_cols}")
        elif fill_numeric == "value":
            if fill_numeric_value is None:
                sys.exit("--fill-numeric value requires --fill-numeric-value")
            df[num_cols] = df[num_cols].fillna(fill_numeric_value)
            change_log.append(f"Filled numeric NaNs with constant {fill_numeric_value}: {num_cols}")

    if fill_text is not None:
        obj_cols = df.select_dtypes(include="object").columns.tolist()
        df[obj_cols] = df[obj_cols].fillna(fill_text)
        change_log.append(f"Filled text NaNs with {fill_text!r}: {obj_cols}")

    # Column selection / drop / rename
    if select_cols:
        missing = [c for c in select_cols if c not in df.columns]
        if missing:
            print(f"[warn] select: missing columns ignored: {missing}")
        df = df[[c for c in select_cols if c in df.columns]]
        change_log.append(f"Selected columns: {select_cols}")

    if drop_cols:
        existing = [c for c in drop_cols if c in df.columns]
        missing = [c for c in drop_cols if c not in df.columns]
        df.drop(columns=existing, inplace=True, errors="ignore")
        change_log.append(f"Dropped columns: {existing}")
        if missing:
            change_log.append(f"[warn] drop: columns not found: {missing}")

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        change_log.append(f"Renamed columns: {rename_map}")

    if verbose:
        print("---- SUMMARY AFTER CLEANING ----")
        print(_summary(df))
        print("--------------------------------")

    return df


# --------- report ------------------------------------------------------------

def write_report(report_path: Path, source: Path, output: Path, log: List[str], df: pd.DataFrame) -> None:
    report = []
    report.append(f"# Data Cleaner Report\n")
    report.append(f"- **Input**: `{source.name}`")
    report.append(f"- **Output**: `{output.name}`")
    report.append("")
    report.append("## Actions")
    if log:
        for item in log:
            report.append(f"- {item}")
    else:
        report.append("- (no actions recorded)")
    report.append("")
    report.append("## Dataset Summary (after cleaning)")
    report.append("")
    report.append("```")
    report.append(_summary(df))
    report.append("```")
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"📝 Report written to {report_path!s}")


# --------- CLI ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cleaner",
        description="Data Cleaner CLI — clean and normalize CSV files"
    )
    p.add_argument("input", help="Path to input CSV file")
    p.add_argument("output", help="Path to save cleaned CSV file")

    # toggles
    p.add_argument("--trim", action="store_true", help="Trim whitespace on all string cells")
    p.add_argument("--drop-blank-rows", action="store_true", help="Drop rows where all values are NaN/empty")
    p.add_argument("--dedupe", action="store_true", help="Drop duplicate rows")
    p.add_argument("-v", "--verbose", action="store_true", help="Print a dataset summary after cleaning")

    # fill strategies
    p.add_argument("--fill-text", default=None, help="Fill NaNs in object/text columns with this value (e.g. 'Unknown')")
    p.add_argument("--fill-numeric", choices=["mean", "median", "value"], help="Strategy to fill numeric NaNs")
    p.add_argument("--fill-numeric-value", type=float, help="Constant value for --fill-numeric value")

    # coercions & normalizations
    p.add_argument("--coerce-numeric", help="Comma/semicolon separated numeric column names to coerce")
    p.add_argument("--coerce-date", help="Comma/semicolon separated date column names to coerce")
    p.add_argument("--dayfirst", action="store_true", help="Interpret day-first dates when coercing")
    p.add_argument("--email-cols", help="Comma/semicolon separated email column names to normalize")

    # column ops
    p.add_argument("--select", help="Keep only these columns (comma/semicolon separated)")
    p.add_argument("--drop", help="Drop these columns (comma/semicolon separated)")
    p.add_argument("--rename", action="append", help="Rename columns with pairs: old=new (repeatable)")

    # report
    p.add_argument("--report", help="Path to write a Markdown report (e.g. reports/clean_report.md)")
    
    # ✅ Added stats aption
    p.add_argument("--stats", action="store_true", help="Show detailed dataset statistics (via pandas.describe)")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    src = Path(args.input)
    dst = Path(args.output)
    _exists_or_die(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # parse lists / mappings
    select_cols = _parse_csv_list(args.select)
    drop_cols = _parse_csv_list(args.drop)
    coerce_numeric_cols = _parse_csv_list(args.coerce_numeric)
    coerce_date_cols = _parse_csv_list(args.coerce_date)
    email_cols = _parse_csv_list(args.email_cols)
    rename_map = _parse_mapping(args.rename)

    # read & clean
    try:
        df = pd.read_csv(src)
    except Exception as e:
        sys.exit(f"Error reading CSV: {e}")

    change_log: List[str] = []

    df = clean_dataframe(
        df,
        trim=args.trim,
        drop_blank_rows=args.drop_blank_rows,
        dedupe=args.dedupe,
        fill_text=args.fill_text,
        fill_numeric=args.fill_numeric,
        fill_numeric_value=args.fill_numeric_value,
        coerce_numeric_cols=coerce_numeric_cols,
        coerce_date_cols=coerce_date_cols,
        dayfirst=args.dayfirst,
        email_cols=email_cols,
        select_cols=select_cols,
        drop_cols=drop_cols,
        rename_map=rename_map,
        verbose=args.verbose,
        change_log=change_log,
    )

    # save
    try:
        df.to_csv(dst, index=False)
    except Exception as e:
        sys.exit(f"Error writing CSV: {e}")
    print(f"✅ Cleaned CSV saved to {dst!s}")

    # report
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_report(report_path, src, dst, change_log, df)


if __name__ == "__main__":
    main()
