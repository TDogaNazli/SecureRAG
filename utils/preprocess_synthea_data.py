#!/usr/bin/env python3


import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime
import gc  # Garbage collector for memory management


def setup_logging(verbose=False):
   """Initialize the logging settings."""
   log_level = logging.DEBUG if verbose else logging.INFO
   logging.basicConfig(
       level=log_level,
       format='%(asctime)s - %(levelname)s - %(message)s'
   )
   return logging.getLogger(__name__)


def read_csv_files(csv_dir):
   """Load CSVs from a directory."""
   dfs = {}
   for path in Path(csv_dir).rglob('*.csv'):
       key = path.stem
       try:
           df = pd.read_csv(path)
           dfs[key] = df
           logging.info(f"Loaded {key}.csv ({len(df)} rows, {df.columns.tolist()})")
       except Exception as err:
           logging.error(f"Failed to read {key}.csv: {err}")
   return dfs


def inspect_dataframes(dataframes, sample_rows=5):
   """Print basic info + sample rows."""
   for name, df in dataframes.items():
       logging.info(f"\nTable: {name}")
       logging.info(f"Shape: {df.shape}")
       logging.info(f"Columns: {df.columns.tolist()}")
       logging.info(f"Head:\n{df.head(sample_rows)}")
       id_candidates = [col for col in df.columns if 'patient' in col.lower()]
       if id_candidates:
           logging.info(f"Potential ID cols: {id_candidates}")


def identify_relationships(dataframes):
   """Stub for linking tables."""
   return {}  # No implementation needed currently


def group_by_patient(dataframes):
   """Group tables by patient ID."""
   output = {
       'grouped_dfs': {},
       'patient_id_columns': {}
   }


   known = {
       'patients': 'Id',
       'conditions': 'PATIENT',
       'medications': 'PATIENT',
       'encounters': 'PATIENT',
       'observations': 'PATIENT',
       'procedures': 'PATIENT',
       'immunizations': 'PATIENT',
       'careplans': 'PATIENT',
       'allergies': 'PATIENT',
       'devices': 'PATIENT',
   }


   for name, df in dataframes.items():
       if name not in known:
           candidates = [col for col in df.columns if 'patient' in col.lower()]
           if candidates:
               known[name] = candidates[0]


   output['patient_id_columns'] = known


   for name, df in dataframes.items():
       pid_col = known.get(name)
       if pid_col and pid_col in df.columns:
           grouped = df.groupby(pid_col)
           output['grouped_dfs'][name] = grouped
           logging.info(f"{name} grouped by {pid_col}: {grouped.ngroups} groups")


   return output


def process_tables_without_direct_patient_id(dataframes, grouped_result, links):
   """Skip admin tables, return grouped result."""
   logging.info("Administrative tables skipped as requested")
   return grouped_result


def join_patient_data(dataframes, grouped_result):
   """Merge all patient data into one DataFrame."""
   grouped_dfs = grouped_result['grouped_dfs']
   pid_columns = grouped_result['patient_id_columns']


   # Use 'patients' as base if available
   if 'patients' in dataframes:
       base = dataframes['patients'].copy()
       pid_col = 'Id'
   else:
       largest = max(
           [(k, v) for k, v in dataframes.items() if any('patient' in c.lower() for c in v.columns)],
           key=lambda x: len(x[1])
       )
       base = largest[1].copy()
       pid_col = next(c for c in base.columns if 'patient' in c.lower())
       logging.warning(f"No patients table, using {largest[0]} with ID column {pid_col}")


   all_pids = set(base[pid_col])


   # Order of processing
   process_order = [
       'patients', 'encounters', 'conditions', 'observations', 'procedures',
       'medications', 'immunizations', 'allergies', 'devices', 'careplans'
   ]
   process_order = [t for t in process_order if t in grouped_dfs or t == 'patients']
   for tbl in grouped_dfs:
       if tbl not in process_order:
           process_order.append(tbl)


   logging.info(f"Processing in order: {process_order}")


   # Identify date fields
   date_fields = {
       (tbl, col)
       for tbl, df in dataframes.items()
       for col in df.columns
       if col.upper() in ('START', 'STOP', 'DATE', 'BIRTHDATE', 'DEATHDATE') or 'DATE' in col.upper()
   }


   unified = []
   total = len(all_pids)


   for idx, pid in enumerate(all_pids):
       if idx % 20 == 0:
           logging.info(f"Processing {idx+1}/{total}")


       record = {'patient_id': pid}


       for tbl in process_order:
           if tbl == 'patients' and 'patients' in dataframes:
               row = dataframes['patients'][dataframes['patients']['Id'] == pid]
               if not row.empty:
                   for col in row.columns:
                       if col == 'Id':
                           continue
                       val = row[col].iloc[0]
                       key = f'demographic_{col}'
                       if ('patients', col) in date_fields and pd.notna(val):
                           try:
                               val = pd.to_datetime(val).isoformat()
                           except:
                               pass
                       record[key] = val if pd.notna(val) else None
               continue


           if tbl not in grouped_dfs:
               continue


           patient_group = None
           pid_field = pid_columns.get(tbl)
           if pid_field:
               try:
                   patient_group = grouped_dfs[tbl].get_group(pid)
               except KeyError:
                   record[f'{tbl}_count'] = 0
                   continue


           if patient_group is None:
               continue


           record[f'{tbl}_count'] = len(patient_group)


           cols = [c for c in patient_group.columns if c != pid_field and c != 'PATIENT_IDS']
           cols.sort()


           # Handle observations
           if tbl == 'observations':
               seen = set()
               obs_data = []
               for _, row in patient_group.iterrows():
                   desc = row.get('DESCRIPTION')
                   val = row.get('VALUE')
                   if pd.notna(desc):
                       key = (str(desc), str(val) if pd.notna(val) else 'None')
                       if key in seen:
                           continue
                       seen.add(key)
                       obs_data.append({
                           'description': desc,
                           'value': val if pd.notna(val) else None,
                           'type': row.get('TYPE') if pd.notna(row.get('TYPE')) else None,
                           'units': row.get('UNITS') if pd.notna(row.get('UNITS')) else None
                       })
               record['observations'] = obs_data


           elif tbl in ['conditions', 'medications', 'procedures', 'immunizations', 'allergies', 'devices', 'careplans']:
               seen = set()
               items = []
               for _, row in patient_group.iterrows():
                   desc = row.get('DESCRIPTION')
                   if pd.notna(desc) and desc not in seen:
                       seen.add(desc)
                       items.append(desc)
               if items:
                   record[tbl] = items


           elif tbl == 'encounters':
               types = set()
               classes = set()
               for _, row in patient_group.iterrows():
                   if pd.notna(row.get('DESCRIPTION')):
                       types.add(row['DESCRIPTION'])
                   if pd.notna(row.get('ENCOUNTERCLASS')):
                       classes.add(row['ENCOUNTERCLASS'])
               if types:
                   record['encounters'] = list(types)
               if classes:
                   record['encounter_classes'] = list(classes)


       unified.append(record)


       if idx % 50 == 0 and idx > 0:
           gc.collect()


   logging.info("Finalizing unified data...")
   final_df = pd.DataFrame(unified)
   size_mb = final_df.memory_usage(deep=True).sum() / (1024 * 1024)
   logging.info(f"Final size: {size_mb:.2f} MB")


   return final_df


def include_ungrouped_tables(unified_df, dataframes, grouped_dfs):
   """Skip tables without patient linkage, focusing on patient health data only."""
   ungrouped = [name for name in dataframes if name not in grouped_dfs and name != 'patients']
  
   if ungrouped:
       logging.warning(f"Skipping unlinked tables: {ungrouped}")
  
   return unified_df


def main():
   parser = argparse.ArgumentParser(description='Preprocess Synthea dataset')
   parser.add_argument('--input-dir', type=str, default='dataset/synthea-dataset-100/set100/csv')
   parser.add_argument('--output-file', type=str, default='dataset/synthea-unified.parquet')
   parser.add_argument('--verbose', action='store_true')
   parser.add_argument('--info-file', type=str, default='')
   parser.add_argument('--exclude-admin', action='store_true', default=True)
   parser.add_argument('--sample', type=int, default=0)
   args = parser.parse_args()


   start_time = datetime.now()
   logger = setup_logging(args.verbose)


   input_dir = os.path.abspath(args.input_dir)
   output_file = os.path.abspath(args.output_file)
   os.makedirs(os.path.dirname(output_file), exist_ok=True)


   logger.info(f"Reading CSV files from {input_dir}...")
   dataframes = read_csv_files(input_dir)


   if not dataframes:
       logger.error("No CSV files found or readable.")
       return


   if args.exclude_admin:
       for table in ['payers', 'organizations', 'providers']:
           if table in dataframes:
               logger.info(f"Excluding admin table: {table}")
               dataframes.pop(table)


   for name, df in dataframes.items():
       date_cols = [col for col in df.columns if col.upper() in ('START', 'STOP', 'DATE', 'BIRTHDATE', 'DEATHDATE')]
       if date_cols:
           logger.info(f"Date columns detected in {name}: {date_cols}")


   logger.info(f"{len(dataframes)} tables after filtering.")


   if args.verbose:
       inspect_dataframes(dataframes)


   logger.info("Identifying relationships...")
   relationships = identify_relationships(dataframes)


   logger.info("Grouping by patient...")
   group_result = group_by_patient(dataframes)


   logger.info("Handling indirect patient links...")
   group_result = process_tables_without_direct_patient_id(dataframes, group_result, relationships)


   logger.info("Merging patient data...")
   unified_df = join_patient_data(dataframes, group_result)


   logger.info("Adding global features...")
   unified_df = include_ungrouped_tables(unified_df, dataframes, group_result['grouped_dfs'])


   logger.info(f"Final dataset: {len(unified_df)} rows, {len(unified_df.columns)} columns.")
   if args.verbose:
       logger.info(f"Columns: {unified_df.columns.tolist()}")


   if args.info_file:
       info_path = os.path.abspath(args.info_file)
       logger.info(f"Saving metadata to {info_path}")


       dataset_info = {
           'num_patients': len(unified_df),
           'num_columns': len(unified_df.columns),
           'columns': {
               table: [col for col in unified_df.columns if col.startswith(f"{table}_") or
                       (table == 'patients' and col.startswith("demographic_"))]
               for table in list(dataframes.keys()) + ['patient']
           },
           'tables': {
               name: {
                   'row_count': len(df),
                   'column_count': len(df.columns),
                   'columns': df.columns.tolist()
               } for name, df in dataframes.items()
           }
       }


       with open(info_path, 'w') as f:
           json.dump(dataset_info, f, indent=2)


   nulls = [col for col in unified_df.columns if unified_df[col].isna().all()]
   if nulls:
       logger.info(f"Dropping {len(nulls)} null-only columns.")
       unified_df = unified_df.drop(columns=nulls)


   logger.info(f"Saving to {output_file}...")
   unified_df.to_parquet(output_file, index=False, compression='snappy')
   logger.info(f"Saved unified data: {len(unified_df)} patients, {len(unified_df.columns)} columns.")


   if args.info_file:
       with open(args.info_file, 'w') as f:
           json.dump({
               'patient_count': len(unified_df),
               'column_count': len(unified_df.columns),
               'columns': unified_df.columns.tolist(),
               'memory_usage_mb': unified_df.memory_usage(deep=True).sum() / (1024 * 1024),
               'processing_time': str(datetime.now() - start_time)
           }, f, indent=2)
       logger.info(f"Dataset info saved to {args.info_file}")


if __name__ == '__main__':
   main()