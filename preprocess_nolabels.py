import pyreadstat
import pandas as pd

df, meta = pyreadstat.read_sav("ATP_W119.sav")

variables = [
    'AIHCCOMF_W119', 'MEDBIAS_W119', 'HCMEDBIAS_W119', 'SCDETECT1_W119', 'SCDETECT2_W119', 'SCDETECT3_W119', 'SCDETECT4_W119', 'AIMH1_W119',       
    'AIMH2_W119', 'AIMH3_W119', 'AIMH5_W119', 'AIPAIN1_W119', 'AIPAIN2_W119', 'AIPAIN3_W119', 'AIPAIN4_W119', 'SROBOT1_W119', 'SROBOT2_W119',     
    'SROBOT3_W119', 'AI_HEARD_W119', 'F_AGECAT', 'F_GENDER', 'F_RACETHNMOD', 'AIKNOW1_W119', 'AIKNOW2_W119', 'AIKNOW3_W119', 'AIKNOW5_W119',
    'AIKNOW6_W119', 'AIKNOW7_W119', 'F_EDUCCAT', 'F_INC_TIER2', 'AIHCTRT1_W119', 'AIWRKH4_W119', 'AIWRK2_a_W119', 'HIREBIAS1_W119',
    'DESRISK_COMF_W119', 'DESRISK_CREAT_W119', 'DESRISK_NTECH_W119', 'RISK2_W119', 'USEAI_W119', 'CNCEXC_W119', 'EVALBIAS1_W119',
    'EMPLSIT_W119', 'INDUSTRYCOMBO_W119', 'F_METRO', 'F_PARTY_FINAL', 'DEVICE_TYPE_W119', 'AIWRK2_b_W119', 'AIWRK2_b_W119',
    'AIWRK2_c_W119', 'AIWRK3_a_W119', 'AIWRK3_b_W119', 'AIWRK3_c_W119', 'AIWRKH1_W119', 'AIWRKH2_a_W119', 'AIWRKH2_b_W119', 
    'AIWRKH3_a_W119', 'AIWRKH3_b_W119', 'AIWRKH3_c_W119', 'AIWRKH3_d_W119', 'AIWRKM1_W119', 'AIWRKM2_a_W119', 'AIWRKM2_b_W119',
    'AIWRKM2_c_W119', 'AIWRKM2_d_W119', 'AIWRKM2_e_W119', 'AIWRKM2_f_W119', 'AIWRKM3_a_W119', 'AIWRKM3_b_W119', 'AIWRKM3_c_W119', 
    'AIWRKM3_d_W119', 'AIWRKM3_e_W119', 'AIWRKM3_f_W119', 'AIWRKM4_a_W119', 'AIWRKM4_b_W119'
]

missing_vars = [var for var in variables if var not in df.columns]
if missing_vars:
    print(f"Warning: Missing variables {missing_vars}. Adjust your list.")
else:
    df_subset = df[variables].copy()

df_clean = df_subset.dropna(subset=['AIHCCOMF_W119'])
df_clean = df_clean[~df_clean.isin([99]).any(axis=1)]
df_clean = df_clean[~df_clean.isin([9.0]).any(axis=1)]

# Save preprocessed data
df_clean.to_csv("W119preprocessed99.csv", index=False)