import streamlit as st
import streamlit.components.v1 as components
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from icalendar import Calendar
from sentence_transformers import SentenceTransformer

# ==========================================
# 1. PAGE CONFIGURATION & CACHING
# ==========================================
st.set_page_config(page_title="Boston AI Week | RL Scheduler", layout="wide")

@st.cache_resource
def load_encoder():
    """Caches the heavy sentence transformer model so it only loads once."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_and_parse_ics(file_path):
    """Caches the parsed calendar data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        cal = Calendar.from_ical(f.read())
        
    events = []
    for component in cal.walk('vevent'):
        summary = str(component.get('summary', ''))
        description = str(component.get('description', ''))
        location_text = str(component.get('location', '')).lower()
        
        # Geocoding Approximation
        coords, area = (42.3555, -71.0601), "Downtown Boston"
        if 'cambridge' in location_text or 'mit' in location_text:
            coords, area = (42.3601, -71.0942), "Cambridge"
        elif 'seaport' in location_text or 'fan pier' in location_text:
            coords, area = (42.3519, -71.0466), "Boston Seaport"
        elif 'burlington' in location_text:
            coords, area = (42.5048, -71.1956), "Burlington"
        elif 'needham' in location_text:
            coords, area = (42.2809, -71.2378), "Needham"
            
        text_content = (summary + " " + description).lower()
        events.append({
            'uid': str(component.get('uid')),
            'summary': summary,
            'description': description,
            'start_time': component.get('dtstart').dt,
            'end_time': component.get('dtend').dt,
            'coords': coords,
            'area': area,
            'is_invite_only': any(kw in text_content for kw in ['invite-only', 'private']),
            'is_free': 'free' in text_content or 'no cost' in text_content
        })
    return events

encoder = load_encoder()
try:
    all_events = load_and_parse_ics("boston-ai-week-2026-2026-08-20_2.ics")
except FileNotFoundError:
    st.error("Error: Make sure 'boston-ai-week-2026-2026-08-20_2.ics' is in the repository.")
    st.stop()

# ==========================================
# 2. RL ENVIRONMENT & MATH
# ==========================================
def haversine(coord1, coord2):
    R = 3958.8 
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2 - lat1)/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1)/2.0)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

class DualEncoderRetriever:
    def rank_candidates(self, user_text, masked_events):
        u = encoder.encode(user_text, normalize_embeddings=True)
        for ev in masked_events:
            v = encoder.encode(ev['summary'] + " " + ev['description'], normalize_embeddings=True)
            ev['score'] = float(np.dot(u, v))
        return sorted(masked_events, key=lambda x: x['score'], reverse=True)

retriever = DualEncoderRetriever()

class ItineraryEnv(gym.Env):
    def __init__(self, candidate_events, alpha=0.1, beta=10.0, max_events=10):
        super(ItineraryEnv, self).__init__()
        self.candidate_events = candidate_events
        self.alpha, self.beta, self.max_events = alpha, beta, max_events
        self.current_schedule = []
        self.available_actions = list(range(len(candidate_events)))
        
    def step(self, action):
        if action not in self.available_actions: return self.current_schedule, -10.0, True, False, {}
        ev = self.candidate_events[action]
        
        # Transit Penalty
        transit = 0.0
        if self.current_schedule and self.current_schedule[-1]['start_time'].date() == ev['start_time'].date():
            transit = self.alpha * haversine(self.current_schedule[-1]['coords'], ev['coords'])
            
        # Overlap Penalty
        overlap = sum(self.beta for s in self.current_schedule if max(s['start_time'], ev['start_time']) < min(s['end_time'], ev['end_time']))
        
        reward = ev['score'] - transit - overlap
        if reward > 0:
            self.current_schedule.append(ev)
            self.current_schedule.sort(key=lambda x: x['start_time'])
            
        self.available_actions.remove(action)
        terminated = len(self.available_actions) == 0 or len(self.current_schedule) >= self.max_events
        return self.current_schedule, reward, terminated, False, {}

    def reset(self, seed=None):
        self.current_schedule = []
        self.available_actions = list(range(len(self.candidate_events)))
        return self.current_schedule, {}

def apply_ui_mask(events, user_filters):
    return [ev for ev in events if not (user_filters.get('invite_only') == False and ev['is_invite_only']) 
            and not (user_filters.get('free_only') == True and not ev['is_free'])]

def generate_optimal_itinerary(persona, filters, events_pool):
    valid_candidates = apply_ui_mask(events_pool, filters)
    ranked_candidates = retriever.rank_candidates(persona, valid_candidates)
    
    env = ItineraryEnv(ranked_candidates, max_events=10)
    state, _ = env.reset()
    for action in range(len(ranked_candidates)):
        state, reward, terminated, _, _ = env.step(action)
        if terminated: break
            
    primary_schedule = state
    suggested = [ev for ev in ranked_candidates if ev['uid'] not in {s['uid'] for s in primary_schedule}][:max(0, 10 - len(primary_schedule))]
    return primary_schedule, suggested

# ==========================================
# 3. STREAMLIT UI & HTML RENDERER
# ==========================================
def render_calendar_html(schedule, suggestions, user_id, persona, filters):
    unique_dates = sorted(list(set(ev['start_time'].date() for ev in schedule))) if schedule else []
    day_start_hour, day_end_hour, hour_height_px = 8, 22, 52
    grid_height_px = (day_end_hour - day_start_hour) * hour_height_px
    card_colors = ["#1a73e8", "#6264A7", "#0078D4", "#0b8043", "#8e24aa", "#d93025"]
    
    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, Roboto, sans-serif; border: 1px solid #dadce0; border-radius: 10px; background: #fff; box-shadow: 0 1px 3px rgba(60,64,67,0.15); overflow: hidden;">
        <div style="background: #f8f9fa; border-bottom: 1px solid #dadce0; padding: 14px 20px;">
            <h3 style="margin: 0; color: #202124; font-size: 16px;">📅 {user_id}</h3>
            <div style="color: #5f6368; font-size: 12px; margin-top: 4px;">{persona}</div>
        </div>
    """
    
    if schedule:
        html += f"""<div style="overflow-x: auto; padding: 12px 16px;"><div style="display: flex; min-width: {len(unique_dates) * 200 + 70}px;">
            <div style="width: 60px; padding-top: 40px; position: relative;">"""
        for h in range(day_start_hour, day_end_hour + 1):
            html += f'<div style="position: absolute; top: {40 + (h - day_start_hour) * hour_height_px - 7}px; right: 8px; font-size: 11px; color: #70757a;">{12 if h%12==0 else h%12} {"AM" if h<12 else "PM"}</div>'
        html += '</div><div style="display: flex; flex-grow: 1; border-left: 1px solid #e0e0e0;">'
        
        for cur_date in unique_dates:
            html += f'<div style="flex: 1; min-width: 190px; border-right: 1px solid #e0e0e0;"><div style="height: 38px; border-bottom: 1px solid #e0e0e0; text-align: center; font-weight: 600; font-size: 13px; color: #3c4043; line-height: 38px; background: #fafafa;">{cur_date.strftime("%a, %b %d")}</div><div style="position: relative; height: {grid_height_px}px;">'
            for h in range(day_start_hour, day_end_hour):
                html += f'<div style="position: absolute; top: {(h - day_start_hour) * hour_height_px}px; left: 0; right: 0; height: {hour_height_px}px; border-bottom: 1px solid #f1f3f4;"></div>'
            
            day_events = [ev for ev in schedule if ev['start_time'].date() == cur_date]
            for ev_idx, ev in enumerate(day_events):
                st, et = max(ev['start_time'].hour + ev['start_time'].minute/60.0, day_start_hour), min(ev['end_time'].hour + ev['end_time'].minute/60.0, day_end_hour)
                html += f"""
                <div style="position: absolute; top: {(st - day_start_hour) * hour_height_px + 2}px; left: 4px; right: 4px; height: {max(et - st, 0.5) * hour_height_px - 4}px; background: {card_colors[ev_idx % len(card_colors)]}; color: #fff; border-radius: 6px; padding: 6px 8px; font-size: 11.5px; display: flex; flex-direction: column; justify-content: space-between; overflow: hidden;">
                    <div><div style="font-size: 10.5px; opacity: 0.9;">{ev['start_time'].strftime('%I:%M %p').lstrip('0')}</div><div style="font-weight: 600; line-height: 1.2;">{ev['summary']}</div></div>
                    <div style="display: flex; justify-content: space-between; font-size: 9.5px;"><span style="background: rgba(255,255,255,0.25); padding: 2px 4px; border-radius: 4px;">★ {ev['score']:.2f}</span><span>📍 {ev['area']}</span></div>
                </div>"""
            html += "</div></div>"
        html += "</div></div></div>"
    else:
        html += '<div style="padding: 24px; color: #5f6368;">No events passed deterministic filters.</div>'

    if suggestions:
        html += '<div style="padding: 16px 20px; background: #f8f9fa; border-top: 1px solid #dadce0;"><h4 style="margin: 0 0 12px 0; font-size: 14px;">💡 Recommended Alternative Events</h4><div style="display: flex; gap: 12px; overflow-x: auto;">'
        for ev in suggestions:
            html += f'<div style="min-width: 200px; max-width: 200px; background: #fff; border: 1px solid #dadce0; border-radius: 8px; padding: 12px;"><div style="font-size: 11px; font-weight: 600; color: #1a73e8; margin-bottom: 6px;">{ev["start_time"].strftime("%b %d, %H:%M")}</div><div style="font-size: 12.5px; font-weight: 600; line-height: 1.3; margin-bottom: 8px;">{ev["summary"]}</div><div style="display: flex; justify-content: space-between; font-size: 10px; color: #70757a;"><span style="font-weight:600;">★ {ev["score"]:.2f}</span><span>📍 {ev["area"]}</span></div></div>'
        html += "</div></div>"
        
    html += "</div>"
    return html

# ==========================================
# 4. APP LAYOUT
# ==========================================
st.title("🧠 Boston AI Week 2026: RL Itinerary Agent")
st.markdown("This tool utilizes a **Dual-Encoder Retriever** and a **Markov Decision Process (MDP)** to dynamically generate optimized schedules by balancing semantic preferences against geospatial transit penalties and temporal overlap constraints.")

profiles = [
    {
        "id": "1. AI Scientist (Simulation & HCI)",
        "persona": "I am a PhD Candidate in Industrial Engineering focusing on machine learning, deep learning, LLMs, and reinforcement learning. I am looking for events bridging simulation and human-computer interaction to transition into an AI Scientist role.",
        "filters": {"invite_only": False, "free_only": False}
    },
    {
        "id": "2. Enterprise CIO",
        "persona": "I am a Chief Information Officer at a large enterprise. I care about cloud infrastructure, cybersecurity, AI governance, enterprise deployment, and high-level strategy.",
        "filters": {"invite_only": True, "free_only": False}
    },
    {
        "id": "3. Junior Developer (Budget)",
        "persona": "I am a student and junior developer. I love coding, building AI agents, hackathons, and open source. I am on a budget so I only want free events.",
        "filters": {"invite_only": False, "free_only": True}
    },
    {
        "id": "4. Startup Founder (B2B)",
        "persona": "I am a startup founder building an AI company. I want to learn about go-to-market strategies, scaling, entrepreneurial finance, meeting VCs, and pitching.",
        "filters": {"invite_only": False, "free_only": False}
    }
]

st.sidebar.header("Agent Parameters")
selected_profile_name = st.sidebar.selectbox("Select User Persona:", [p['id'] for p in profiles])
selected_profile = next(p for p in profiles if p['id'] == selected_profile_name)

st.sidebar.markdown("### Deterministic UI Masks")
st.sidebar.toggle("Strict: Free Events Only", value=selected_profile['filters']['free_only'], disabled=True)
st.sidebar.toggle("Access: Invite-Only Allowed", value=selected_profile['filters']['invite_only'], disabled=True)

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Generate Itinerary", type="primary", use_container_width=True):
    with st.spinner("Initializing RL Environment & Computing Vectors..."):
        primary, suggested = generate_optimal_itinerary(selected_profile['persona'], selected_profile['filters'], all_events)
        html_output = render_calendar_html(primary, suggested, selected_profile['id'], selected_profile['persona'], selected_profile['filters'])
        components.html(html_output, height=850, scrolling=True)
else:
    st.info("👈 Select a persona from the sidebar and click **Generate Itinerary** to simulate the RL agent.")