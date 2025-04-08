# 1. Skin Cancer Detection (SCDETECT1-4)
sc_awareness = df['SCDETECT1_W119'].value_counts(normalize=True).mul(100)
sc_advance = df['SCDETECT2_W119'].value_counts(normalize=True).mul(100)
sc_willingness = df['SCDETECT3_W119'].value_counts(normalize=True).mul(100)
sc_accuracy = df['SCDETECT4_W119'].value_counts(normalize=True).mul(100)

# 2. Mental Health Chatbots (AIMH1-5)
mh_awareness = df['AIMH1_W119'].value_counts(normalize=True).mul(100)
mh_advance = df['AIMH2_W119'].value_counts(normalize=True).mul(100)
mh_willingness = df['AIMH3_W119'].value_counts(normalize=True).mul(100)
mh_condition = df['AIMH5_W119'].value_counts(normalize=True).mul(100)  # Usage conditions

# 3. Pain Medication (AIPAIN1-4)
pain_awareness = df['AIPAIN1_W119'].value_counts(normalize=True).mul(100)
pain_advance = df['AIPAIN2_W119'].value_counts(normalize=True).mul(100)
pain_willingness = df['AIPAIN3_W119'].value_counts(normalize=True).mul(100)
pain_impact = df['AIPAIN4_W119'].value_counts(normalize=True).mul(100)

# 4. Surgical Robots (SROBOT1-3)
robot_awareness = df['SROBOT1_W119'].value_counts(normalize=True).mul(100)
robot_advance = df['SROBOT2_W119'].value_counts(normalize=True).mul(100)
robot_willingness = df['SROBOT3_W119'].value_counts(normalize=True).mul(100)

def plot_metrics(app_name, metrics_dict, figsize=(15,4)):
    fig, axs = plt.subplots(1, len(metrics_dict), figsize=figsize)
    fig.suptitle(f"{app_name} Public Acceptance Metrics", fontsize=14)
    
    for ax, (title, data) in zip(axs, metrics_dict.items()):
        data.plot(kind='bar', ax=ax, color='#1f77b4')
        ax.set_title(title)
        ax.set_ylabel("% of Respondents")
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

plot_metrics("Skin Cancer AI", {
    "Awareness": sc_awareness,
    "Perceived Advancement": sc_advance,
    "Willingness to Use": sc_willingness,
    "Perceived Accuracy": sc_accuracy
})

plot_metrics("Surgical Robots", {
    "Awareness": robot_awareness,
    "Perceived Advancement": robot_advance,
    "Willingness to Use": robot_willingness
})

plot_metrics("Mental Health Chatbots", {
    "Awareness": mh_awareness,
    "Condition": mh_condition,
    "Perceived Advancement": mh_advance,
    "Willingness to Use": mh_willingness
})

plot_metrics("Pain Management AI", {
    "Awareness": pain_awareness,
    "Pain impact": pain_impact,
    "Perceived Advancement": pain_advance,
    "Willingness to Use": pain_willingness
})