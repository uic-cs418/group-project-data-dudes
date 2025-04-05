import pyreadstat
import pandas as pd

df, meta = pyreadstat.read_sav("ATP_W119.sav")

variables = [
    'AIHCCOMF_W119', 'MEDBIAS_W119', 'HCMEDBIAS_W119', 'SCDETECT1_W119', 'SCDETECT2_W119', 'SCDETECT3_W119', 'SCDETECT4_W119', 'AIMH1_W119',       
    'AIMH2_W119', 'AIMH3_W119', 'AIMH5_W119', 'AIPAIN1_W119', 'AIPAIN2_W119', 'AIPAIN3_W119', 'AIPAIN4_W119', 'SROBOT1_W119', 'SROBOT2_W119',     # Advance level of surgical robots
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

response_maps = {
    'AIHCCOMF_W119': {
        1: "Very comfortable",
        2: "Somewhat comfortable",
        3: "Somewhat uncomfortable",
        4: "Very uncomfortable"
    },
    'MEDBIAS_W119': {
        1: "Major problem",
        2: "Minor problem",
        3: "Not a problem"
    },
    'HCMEDBIAS_W119': {
        1: "Definitely get better",
        2: "Probably get better",
        3: "Stay the same",
        4: "Probably get worse",
        5: "Definitely get worse"
    },
    'SCDETECT1_W119': {
        1: "A lot",
        2: "A little",
        3: "Nothing"
    },
    'SCDETECT2_W119': {
        1: "Major advance",
        2: "Minor advance",
        3: "Not an advance",
        9: "Not sure"
    },
    'SCDETECT3_W119': {
        1: "Definitely want",
        2: "Probably want",
        3: "Probably not",
        4: "Definitely not"
    },
    'SCDETECT4_W119': {
        1: "More accurate",
        2: "Less accurate",
        3: "Not make much difference"
    },
    'F_AGECAT': {
        1: "18-29",
        2: "30-49",
        3: "50-64",
        4: "65+"
    },
    'AI_HEARD_W119': {
        1: "A lot",
        2: "A little",
        3: "Nothing"
    },
    'F_GENDER' : {
        1: "Male",
        2: "Female"
    },
        'SROBOT1_W119': {  
        2: "A little",
        3: "Nothing"
    },
    'SROBOT2_W119': {  
        1: "Major advance",
        2: "Minor advance",
        3: "Not an advance",
        9: "Not sure"
    },
    'SROBOT3_W119': {   
        1: "Definitely want",
        2: "Probably want",
        3: "Probably NOT want",
        4: "Definitely NOT want"
    },
        'AIPAIN1_W119': {   
        1: "A lot",
        2: "A little",
        3: "Nothing"
    },
    'AIPAIN2_W119': {  
        1: "Major advance",
        2: "Minor advance",
        3: "Not an advance",
        9: "Not sure"
    },
    'AIPAIN3_W119': {   
        1: "Definitely want",
        2: "Probably want",
        3: "Probably NOT want",
        4: "Definitely NOT want"
    },
    'AIMH1_W119': {   
        1: "A lot",
        2: "A little",
        3: "Nothing"
    },
    'AIMH2_W119': {  
        1: "Major advance",
        2: "Minor advance",
        3: "Not an advance",
        9: "Not sure"
    },
    'AIMH3_W119': {  
        1: "Definitely want",
        2: "Probably want",
        3: "Probably NOT want",
        4: "Definitely NOT want"
    },
    'AIKNOW1_W119': {
        1: "Chatbot answers questions",
        2: "Online survey",
        3: "Contact form",
        4: "FAQ webpage",
        9: "Not sure"
    },
    'AIKNOW2_W119': {
        1: "Bluetooth speakers",
        2: "Playlist recommendation",
        3: "WiFi streaming",
        4: "Shuffle play",
        9: "Not sure"
    },
    'AIKNOW3_W119': {
        1: "Mark email as read",
        2: "Schedule email",
        3: "Categorize spam",  
        4: "Sort by time",
        9: "Not sure"
    },
    'AIKNOW5_W119': {
        1: "Wearable fitness trackers",  
        2: "Thermometers",
        3: "COVID-19 tests",
        4: "Pulse oximeters",
        9: "Not sure"
    },
    'AIKNOW6_W119': {
        1: "Account storage",
        2: "Purchase history",
        3: "Product recommendations",  
        4: "Customer reviews",
        9: "Not sure"
    },
    'AIKNOW7_W119': {
        1: "Program thermostat",
        2: "Security camera alerts",  
        3: "Light timer",
        4: "Water filter light",
        9: "Not sure"
    }
}

for var, mapping in response_maps.items():
    if var in df_clean.columns:
        df_clean[var] = df_clean[var].map(mapping)

df_clean.to_csv("W119preprocessed.csv", index=False)