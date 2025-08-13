import numpy as np
import pandas as pd
import os
import yaml

# =============================
# Environment Setup
# =============================

np.random.seed(42)  # Set a fixed seed for reproducibility

# Project root directory (relative to this file's location)
project_root = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(project_root, 'raw_data')           # Directory containing raw data
output_dir   = os.path.join(project_root, 'generated_traffic')  # Directory for generated data

os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist
print(f"Project Root: {project_root}")

# =============================
# 1. Retrieve List of Raw Data Files
# =============================
def get_all_raw_data_files(raw_data_dir):
    """
    Find all CSV files in raw_data_dir with the suffix '_dataset.csv'.
    Returns a list of absolute file paths.
    """
    raw_files = [os.path.join(raw_data_dir, file) 
                 for file in os.listdir(raw_data_dir) 
                 if file.endswith('_dataset.csv')]
    assert len(raw_files) > 0, "❌ No raw data files found!"
    return raw_files

# =============================
# 2. Load and Aggregate Total DL Bitrate from All Files
# =============================
def load_total_DL(raw_data_files):
    """
    Read each file, identify columns containing 'DL_bitrate', 
    sum vertically across all BS in each file,
    and then sum horizontally across files to obtain total DL traffic per timestep.
    """
    total_dl = None
    for file in raw_data_files:
        df = pd.read_csv(file)
        dl_cols = [col for col in df.columns if 'DL_bitrate' in col]
        assert len(dl_cols) > 0, f"❌ No DL_bitrate columns found in {file}"
        
        # Total traffic from all BS in this file for each timestep
        dl_sum = df[dl_cols].sum(axis=1)

        # Accumulate across files
        if total_dl is None:
            total_dl = dl_sum
        else:
            total_dl += dl_sum
    return total_dl.values

# =============================
# 3. Generate traffic.csv
# =============================
def generate_traffic(raw_data_dir, output_dir, grid_rows, grid_cols):
    """
    Generate traffic data for the entire BS grid over time,
    with dynamic hotspots/coldspots, daily patterns, and event spikes.
    """
    raw_data_files = get_all_raw_data_files(raw_data_dir)
    total_dl = load_total_DL(raw_data_files)

    # Trim data after first NaN occurrence
    nan_index = np.where(np.isnan(total_dl))[0]
    if len(nan_index) > 0:
        cut_point = nan_index[0]
        print(f"NaN detected at timestep {cut_point}, trimming data.")
        total_dl = total_dl[:cut_point]
    max_timesteps = len(total_dl)

    n_bs = grid_rows * grid_cols

    # Initialize base "hotness" for each BS
    base_hotness = np.random.uniform(0.5, 2.0, n_bs)
    hotspot_indices = np.random.choice(n_bs, size=int(0.3 * n_bs), replace=False)
    for idx in hotspot_indices:
        base_hotness[idx] = np.random.uniform(3.0, 4.0)  # Stronger hotspots

    # Hourly traffic pattern (rush hour, nighttime, normal)
    def daily_pattern(t):
        hour = t % 24
        if 7 <= hour <= 9 or 18 <= hour <= 22:
            return 2.2
        elif 0 <= hour <= 5:
            return 0.5
        else:
            return 1.0

    # Probability of an unusual traffic spike
    def event_spike(prob=0.05):
        return np.random.choice([1.0, np.random.uniform(1.2, 2.0)], p=[1-prob, prob])

    traffic_data = []
    for t in range(max_timesteps):
        row = {'timestep': int(t)}
        hour = t % 24

        dynamic_hotness = np.copy(base_hotness)

        # Hotspots have reduced load at night
        if 0 <= hour <= 5:
            for idx in hotspot_indices:
                dynamic_hotness[idx] *= 0.5

        # Coldspots have a 10% chance of a sudden spike
        coldspot_indices = [i for i in range(n_bs) if i not in hotspot_indices]
        for idx in coldspot_indices:
            if np.random.rand() < 0.1:
                dynamic_hotness[idx] *= np.random.uniform(1.5, 3.0)

        # Softmax normalization so that hotness values sum to 1
        soft_hotness = np.exp(dynamic_hotness) / np.sum(np.exp(dynamic_hotness))

        # Compute traffic for each BS
        for bs_id in range(n_bs):
            base_traffic = total_dl[t] * soft_hotness[bs_id]
            traffic = base_traffic * daily_pattern(t) * event_spike() * (1 + np.random.normal(0, 0.05))
            traffic = max(traffic, 0.0)
            row[f'BS_{bs_id}_DL'] = float(traffic)

        traffic_data.append(row)

    # Save to CSV
    df = pd.DataFrame(traffic_data)
    df.to_csv(os.path.join(output_dir, 'traffic.csv'), index=False)
    print("traffic.csv generated with dynamic hotspots/coldspots")

    return max_timesteps

# =============================
# 4. Generate BS Positions
# =============================
def generate_bs(output_dir, grid_rows, grid_cols, cell_radius):
    """
    Generate coordinates for BS in a grid_rows x grid_cols arrangement.
    Each BS is spaced 2*cell_radius apart in both X and Y directions.
    """
    bs_list = []
    bs_id = 0
    for row in range(grid_rows):
        for col in range(grid_cols):
            x = col * cell_radius * 2
            y = row * cell_radius * 2
            bs_list.append({
                'bs_id': bs_id,
                'x_pos': float(x),
                'y_pos': float(y),
                'cell_radius': float(cell_radius),
                'row': row,
                'col': col
            })
            bs_id += 1

    pd.DataFrame(bs_list).to_csv(os.path.join(output_dir, 'bs_info.csv'), index=False)
    print("bs_info.csv generated")

# =============================
# 5. Generate User Information
# =============================
def generate_users(output_dir, grid_rows, grid_cols, cell_radius, total_users):
    """
    Randomly generate users within the grid area.
    Each user is assigned to the nearest BS.
    User types: voice / data / video (with ratios 8% / 21% / 71%).
    """
    if os.path.exists(os.path.join(output_dir, 'user_info.csv')):
        os.remove(os.path.join(output_dir, 'user_info.csv'))

    bs_pos = [(col * cell_radius * 2, row * cell_radius * 2)
              for row in range(grid_rows)
              for col in range(grid_cols)]
    n_bs = grid_rows * grid_cols

    user_home_cells = []
    for _ in range(total_users):
        x = np.random.uniform(0, grid_cols * cell_radius * 2)
        y = np.random.uniform(0, grid_rows * cell_radius * 2)
        dists = [np.linalg.norm([x - bsx, y - bsy]) for bsx, bsy in bs_pos]
        home_bs = np.argmin(dists)
        user_home_cells.append((x, y, home_bs))

    users_list = []
    for user_id, (x, y, home_bs) in enumerate(user_home_cells):
        user_type = np.random.choice(['voice', 'data', 'video'], p=[0.08, 0.21, 0.71])
        users_list.append({
            'user_id': user_id,
            'x_pos': float(x),
            'y_pos': float(y),
            'home_cell': home_bs,
            'user_type': user_type
        })

    pd.DataFrame(users_list).to_csv(os.path.join(output_dir, 'user_info.csv'), index=False)
    print("user_info.csv generated")

# =============================
# 6. Generate Episodes
# =============================
def generate_episodes(output_dir, grid_rows, grid_cols, traffic_file, max_timesteps_per_episode, max_episodes):
    """
    Split total traffic into multiple episodes.
    Add variability: noise, random hotspot/coldspot changes, and abnormal spikes.
    """
    n_bs = grid_rows * grid_cols
    traffic = pd.read_csv(traffic_file)

    episode_noise_std = 0.05
    spike_prob = 0.05
    spike_mult_range = (1.5, 3.0)
    hotspot_prob = 0.2
    hotspot_mult_range = (1.2, 1.5)
    coldspot_mult_range = (0.5, 0.8)

    bs_means = {bs: traffic[f'BS_{bs}_DL'].mean() for bs in range(n_bs)}

    episodes = []
    episode_traffic_records = []

    for ep in range(max_episodes):
        # Random multiplier for each BS
        bs_multiplier = np.ones(n_bs)
        for bs in range(n_bs):
            r = np.random.rand()
            if r < hotspot_prob / 2:
                bs_multiplier[bs] = np.random.uniform(*hotspot_mult_range)
            elif r < hotspot_prob:
                bs_multiplier[bs] = np.random.uniform(*coldspot_mult_range)

        for t in range(max_timesteps_per_episode):
            global_timestep = (ep * max_timesteps_per_episode + t) % len(traffic)
            bs_status = []
            bs_traffic_this_timestep = []

            for bs in range(n_bs):
                bs_dl = traffic.loc[global_timestep, f'BS_{bs}_DL'] * bs_multiplier[bs] * (1 + np.random.normal(0, episode_noise_std))
                if np.random.rand() < spike_prob:
                    bs_dl *= np.random.uniform(*spike_mult_range)
                bs_traffic_this_timestep.append(bs_dl)
                threshold = 0.5 * bs_means[bs]
                bs_status.append(1 if bs_dl >= threshold else 0)

            episode_traffic_records.append({
                'episode_id': ep,
                'timestep': t,
                **{f'BS_{bs}_DL': bs_traffic_this_timestep[bs] for bs in range(n_bs)}
            })
            episodes.append({
                'episode_id': ep,
                'timestep': t,
                'bs_status': bs_status
            })

    pd.DataFrame(episode_traffic_records).to_csv(os.path.join(output_dir, 'episodes.csv'), index=False)
    print("episodes.csv generated with noise + hotspot/coldspot + spike")
    return pd.DataFrame(episode_traffic_records)

# =============================
# 7. Generate rate_requirements.csv
# =============================
def generate_rate_requirements(output_dir, traffic_df):
    """
    Convert BS traffic into bandwidth requirements for each user 
    based on their assigned BS and service type.
    """
    if os.path.exists(os.path.join(output_dir, 'rate_requirements.csv')):
        os.remove(os.path.join(output_dir, 'rate_requirements.csv'))

    user_df = pd.read_csv(os.path.join(output_dir, 'user_info.csv'))
    user_per_bs = user_df['home_cell'].value_counts().to_dict()

    data = []
    for _, row in traffic_df.iterrows():
        ep = row['episode_id']
        timestep = row['timestep']
        for _, user in user_df.iterrows():
            bs_id = user['home_cell']
            bs_traffic = row[f'BS_{bs_id}_DL']
            n_users = user_per_bs.get(bs_id, 1)
            base_rate = bs_traffic / n_users
            scale_factor = {'voice': 0.2, 'data': 1.0, 'video': 3.0}
            rate_req = base_rate * scale_factor[user['user_type']]
            rate_req_mbps = max(rate_req * 8 / 1e6, 0.1)
            data.append({
                'episode_id': ep,
                'timestep': timestep,
                'user_id': user['user_id'],
                'rate_requirement': rate_req_mbps
            })
    pd.DataFrame(data).to_csv(os.path.join(output_dir, 'rate_requirements.csv'), index=False)
    print("rate_requirements.csv generated")

# =============================
# 8. Generate UAV Home Positions
# =============================
def generate_uav_home_positions(output_dir, grid_rows, grid_cols, total_episodes, total_uavs, cell_radius):
    """
    Generate initial and final UAV positions for each episode.
    Currently, UAVs do not move (x_final = x_init).
    """
    uav_data = []
    home_point = (0, 0)  # Default starting point
    for ep in range(total_episodes):
        for uav_id in range(total_uavs):
            x, y = home_point
            uav_data.append({
                'episode_id': ep,
                'uav_id': uav_id,
                'x_init': x,
                'y_init': y,
                'x_final': x,
                'y_final': y,
            })
    pd.DataFrame(uav_data).to_csv(os.path.join(output_dir, 'uav_home_positions.csv'), index=False)
    print("uav_home_positions.csv generated")

# =============================
# 9. Save Config
# =============================
def save_config(output_dir, config_dict):
    """Save the configuration file in YAML format."""
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False)
    print("✅ config.yaml saved")

# =============================
# Run Full Data Generation Pipeline
# =============================
if __name__ == "__main__":
    print("🔄 Starting data generation...")
    config = {
        'grid_rows': 3,
        'grid_cols': 3,
        'cell_radius': 500,
        'total_users': 100,
        'total_uavs': 2,
        'user_rate_range': [0.5, 5.0],
        'max_episodes': 100,
        'max_timesteps_per_episode': 500
    }

    generate_traffic(raw_data_dir, output_dir, config['grid_rows'], config['grid_cols'])
    generate_bs(output_dir, config['grid_rows'], config['grid_cols'], config['cell_radius'])
    generate_users(output_dir, config['grid_rows'], config['grid_cols'], config['cell_radius'], config['total_users'])
    traffic_df = generate_episodes(output_dir, config['grid_rows'], config['grid_cols'], os.path.join(output_dir, 'traffic.csv'), config['max_timesteps_per_episode'], config['max_episodes'])
    generate_rate_requirements(output_dir, traffic_df)
    generate_uav_home_positions(output_dir, config['grid_rows'], config['grid_cols'], config['max_episodes'], config['total_uavs'], config['cell_radius'])
    save_config(output_dir, config)

    print("✅ Data generation completed successfully!")
