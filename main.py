#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from typing import List, Optional
import pandas as pd
from cleaner import build_parser, _exists_or_die, _parse_csv_list, _parse_mapping, clean_dataframe, write_report

def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    src = Path(args.input)
    dst = Path(args.output)
    _exists_or_die(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # parse lists / mappings
    select_cols: List[str] = _parse_csv_list(args.select)
    drop_cols: List[str] = _parse_csv_list(args.drop)
    coerce_numeric_cols: List[str] = _parse_csv_list(args.coerce_numeric)
    coerce_date_cols: List[str] = _parse_csv_list(args.coerce_date)
    email_cols: List[str] = _parse_csv_list(args.email_cols)
    rename_map = _parse_mapping(args.rename)

    # read CSV
    try:
        df = pd.read_csv(src)
    except Exception as e:
        sys.exit(f"Error reading CSV: {e}")

    change_log: List[str] = []

    # clean dataframe
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

    # save CSV
    try:
        df.to_csv(dst, index=False)
        print(f"✅ Cleaned CSV saved to {dst!s}")
    except Exception as e:
        sys.exit(f"Error writing CSV: {e}")
    

    # write report if requested
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_report(report_path, src, dst, change_log, df)

if __name__ == "__main__":
    main()
