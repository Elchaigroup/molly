import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="SEO Analyzer Pro", layout="wide")

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import time
import os
import json
from datetime import datetime
import sqlite3
from pathlib import Path

# -------------------
# ENV VARIABLES (SECURE)
# -------------------
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

# Check for required environment variables
if not SCRAPINGDOG_API_KEY or not GROQ_API_KEY:
    st.error("Missing API Keys! Please configure SCRAPINGDOG_API_KEY and GROQ_API_KEY in environment variables.")
    st.info("Contact your administrator to set up the required API keys.")
    st.stop()

# -------------------
# DATABASE CONFIGURATION (Future-ready for online DB)
# -------------------
USE_ONLINE_DB = os.getenv("USE_ONLINE_DB", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL")
LOCAL_DB_PATH = "/tmp/seo_analysis_history.db" if os.getenv("RENDER") else "seo_analysis_history.db"

class DatabaseManager:
    """Modular database manager - easy to switch between SQLite and PostgreSQL"""
    
    def __init__(self):
        self.db_type = "postgresql" if USE_ONLINE_DB and DATABASE_URL else "sqlite"
        self.init_database()
    
    def get_connection(self):
        """Get database connection based on configuration"""
        if self.db_type == "postgresql":
            import psycopg2
            return psycopg2.connect(DATABASE_URL, sslmode='require')
        else:
            return sqlite3.connect(LOCAL_DB_PATH)
    
    def init_database(self):
        """Initialize database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if self.db_type == "postgresql":
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    topic TEXT,
                    client_site TEXT,
                    num_sites INTEGER,
                    max_pages INTEGER,
                    ai_recommendations TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS competitor_results (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES analysis_sessions(id) ON DELETE CASCADE,
                    url TEXT,
                    title TEXT,
                    meta TEXT,
                    h1 TEXT,
                    word_count INTEGER,
                    seo_score INTEGER,
                    content TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS client_results (
                    id SERIAL PRIMARY KEY,
                    session_id INTEGER REFERENCES analysis_sessions(id) ON DELETE CASCADE,
                    url TEXT,
                    title TEXT,
                    meta TEXT,
                    h1 TEXT,
                    word_count INTEGER,
                    seo_score INTEGER,
                    content TEXT
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    topic TEXT,
                    client_site TEXT,
                    num_sites INTEGER,
                    max_pages INTEGER,
                    ai_recommendations TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS competitor_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    url TEXT,
                    title TEXT,
                    meta TEXT,
                    h1 TEXT,
                    word_count INTEGER,
                    seo_score INTEGER,
                    content TEXT,
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions(id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS client_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    url TEXT,
                    title TEXT,
                    meta TEXT,
                    h1 TEXT,
                    word_count INTEGER,
                    seo_score INTEGER,
                    content TEXT,
                    FOREIGN KEY (session_id) REFERENCES analysis_sessions(id)
                )
            ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis_session(self, topic, client_site, num_sites, max_pages, ai_recommendations, competitor_data, client_data):
        """Save analysis session to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if self.db_type == "postgresql":
            cursor.execute('''
                INSERT INTO analysis_sessions (topic, client_site, num_sites, max_pages, ai_recommendations)
                VALUES (%s, %s, %s, %s, %s) RETURNING id
            ''', (topic, client_site, num_sites, max_pages, ai_recommendations))
            session_id = cursor.fetchone()[0]
        else:
            cursor.execute('''
                INSERT INTO analysis_sessions (topic, client_site, num_sites, max_pages, ai_recommendations)
                VALUES (?, ?, ?, ?, ?)
            ''', (topic, client_site, num_sites, max_pages, ai_recommendations))
            session_id = cursor.lastrowid
        
        # Insert competitor results
        for result in competitor_data:
            if self.db_type == "postgresql":
                cursor.execute('''
                    INSERT INTO competitor_results 
                    (session_id, url, title, meta, h1, word_count, seo_score, content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (session_id, result['url'], result['title'], result['meta'], 
                      result['h1'], result['word_count'], result['seo_score'], result['content']))
            else:
                cursor.execute('''
                    INSERT INTO competitor_results 
                    (session_id, url, title, meta, h1, word_count, seo_score, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, result['url'], result['title'], result['meta'], 
                      result['h1'], result['word_count'], result['seo_score'], result['content']))
        
        # Insert client results
        for result in client_data:
            if self.db_type == "postgresql":
                cursor.execute('''
                    INSERT INTO client_results 
                    (session_id, url, title, meta, h1, word_count, seo_score, content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (session_id, result['url'], result['title'], result['meta'], 
                      result['h1'], result['word_count'], result['seo_score'], result['content']))
            else:
                cursor.execute('''
                    INSERT INTO client_results 
                    (session_id, url, title, meta, h1, word_count, seo_score, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, result['url'], result['title'], result['meta'], 
                      result['h1'], result['word_count'], result['seo_score'], result['content']))
        
        conn.commit()
        conn.close()
        return session_id
    
    def load_analysis_sessions(self):
        """Load all analysis sessions from database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, timestamp, topic, client_site, num_sites, max_pages 
            FROM analysis_sessions 
            ORDER BY timestamp DESC
        ''')
        sessions = cursor.fetchall()
        conn.close()
        
        # Convert to list of dicts
        session_list = []
        for session in sessions:
            session_list.append({
                'id': session[0],
                'timestamp': session[1],
                'topic': session[2],
                'client_site': session[3],
                'num_sites': session[4],
                'max_pages': session[5]
            })
        
        return session_list
    
    def load_session_details(self, session_id):
        """Load detailed results for a specific session"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Load session info
        param = (session_id,)
        if self.db_type == "postgresql":
            cursor.execute('SELECT * FROM analysis_sessions WHERE id = %s', param)
        else:
            cursor.execute('SELECT * FROM analysis_sessions WHERE id = ?', param)
        
        session_info = cursor.fetchone()
        
        # Load competitor results
        if self.db_type == "postgresql":
            cursor.execute('SELECT * FROM competitor_results WHERE session_id = %s', param)
        else:
            cursor.execute('SELECT * FROM competitor_results WHERE session_id = ?', param)
        
        competitor_results = cursor.fetchall()
        
        # Load client results
        if self.db_type == "postgresql":
            cursor.execute('SELECT * FROM client_results WHERE session_id = %s', param)
        else:
            cursor.execute('SELECT * FROM client_results WHERE session_id = ?', param)
            
        client_results = cursor.fetchall()
        
        conn.close()
        
        # Convert to proper format
        if session_info:
            session_dict = {
                'id': session_info[0],
                'timestamp': session_info[1],
                'topic': session_info[2],
                'client_site': session_info[3],
                'num_sites': session_info[4],
                'max_pages': session_info[5],
                'ai_recommendations': session_info[6],
                'session_id': session_info[0]
            }
            
            # Convert results to list of dicts
            competitor_list = []
            for row in competitor_results:
                competitor_list.append({
                    'url': row[2], 'title': row[3], 'meta': row[4],
                    'h1': row[5], 'word_count': row[6], 'seo_score': row[7], 'content': row[8]
                })
                
            client_list = []
            for row in client_results:
                client_list.append({
                    'url': row[2], 'title': row[3], 'meta': row[4],
                    'h1': row[5], 'word_count': row[6], 'seo_score': row[7], 'content': row[8]
                })
            
            return session_dict, competitor_list, client_list
        
        return None, [], []
    
    def delete_session(self, session_id):
        """Delete a session and all its related data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        param = (session_id,)
        if self.db_type == "postgresql":
            cursor.execute('DELETE FROM competitor_results WHERE session_id = %s', param)
            cursor.execute('DELETE FROM client_results WHERE session_id = %s', param)
            cursor.execute('DELETE FROM analysis_sessions WHERE id = %s', param)
        else:
            cursor.execute('DELETE FROM competitor_results WHERE session_id = ?', param)
            cursor.execute('DELETE FROM client_results WHERE session_id = ?', param)
            cursor.execute('DELETE FROM analysis_sessions WHERE id = ?', param)
        
        conn.commit()
        conn.close()

# Initialize database manager
@st.cache_resource
def get_db_manager():
    return DatabaseManager()

db = get_db_manager()

# -------------------
# HELPER FUNCTIONS
# -------------------
def search_topic(topic, num_results=3):
    """Search for competitor sites using ScrapingDog Google API"""
    url = f"https://api.scrapingdog.com/google?api_key={SCRAPINGDOG_API_KEY}&query={topic}&num={num_results}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            st.warning(f"Google search API failed with status {resp.status_code}")
            return []
        data = resp.json()
        return [item["link"] for item in data.get("organic_results", [])]
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

def fetch_html(url):
    """Fetch with ScrapingDog (for competitor sites)"""
    scrape_url = f"https://api.scrapingdog.com/scrape?api_key={SCRAPINGDOG_API_KEY}&url={url}"
    try:
        resp = requests.get(scrape_url, timeout=60)
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception:
        return None

def fetch_html_client(url):
    """Direct requests for client site"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        return resp.text if resp.status_code == 200 else None
    except Exception as e:
        st.warning(f"Client fetch failed: {e}")
        return None

def extract_links(html, base_url):
    """Extract internal links from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"])
        parsed = urlparse(full)
        if parsed.netloc == urlparse(base_url).netloc:
            norm = parsed.scheme + "://" + parsed.netloc + parsed.path
            links.add(norm.rstrip("/"))
    return links

def analyze_page(html, url, keyword):
    """Analyze a single page for SEO metrics"""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title else ""
    meta = soup.find("meta", {"name": "description"})
    meta_desc = meta["content"].strip() if meta else ""
    h1 = soup.find("h1")
    h1_text = h1.get_text(strip=True) if h1 else ""
    text = soup.get_text(" ", strip=True)
    word_count = len(text.split())

    score = 0
    if keyword.lower() in title.lower(): score += 10
    if keyword.lower() in h1_text.lower(): score += 5
    if keyword.lower() in meta_desc.lower(): score += 5
    if 50 <= len(title) <= 60: score += 2
    if 150 <= len(meta_desc) <= 160: score += 2
    if word_count > 300: score += 3

    return {
        "url": url,
        "title": title,
        "meta": meta_desc,
        "h1": h1_text,
        "word_count": word_count,
        "seo_score": score,
        "content": text[:2000]
    }

def crawl_site(start_url, keyword, max_pages=5, progress_area=None, client=False):
    """Crawl a website and analyze pages for SEO"""
    visited = set()
    queue = deque([start_url.rstrip("/")])
    results = []

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)

        msg = f"Crawling: {url}"
        if progress_area:
            progress_area.write(msg)

        html = fetch_html_client(url) if client else fetch_html(url)
        if not html:
            if progress_area:
                progress_area.write(f"Failed to fetch {url}")
            continue

        results.append(analyze_page(html, url, keyword))

        for link in extract_links(html, start_url):
            if link not in visited and link not in queue:
                queue.append(link)

        time.sleep(1)

    return results

def groq_generate(prompt):
    """Generate AI recommendations using Groq API"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"Groq API Error: {resp.text}"
    except Exception as e:
        return f"Groq API Error: {e}"

# Helper function to create CSV content
def create_csv(data, headers):
    """Create CSV content from list of dicts"""
    if not data:
        return "No data available"
    
    csv_lines = [','.join(headers)]
    for item in data:
        row = []
        for header in headers:
            value = str(item.get(header, '')).replace(',', ';').replace('\n', ' ')
            row.append(f'"{value}"')
        csv_lines.append(','.join(row))
    
    return '\n'.join(csv_lines)

# -------------------
# SESSION STATE
# -------------------
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'show_history' not in st.session_state:
    st.session_state.show_history = False

# -------------------
# STREAMLIT APP
# -------------------

# Header
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.title("SEO Analyzer Pro")
    st.caption("SQLite Database (Pandas-Free Version)")

with col2:
    if st.session_state.show_history:
        if st.button("Back to Analysis", type="secondary"):
            st.session_state.show_history = False
            st.experimental_rerun()
    else:
        if st.button("View History", type="secondary"):
            st.session_state.show_history = True
            st.experimental_rerun()

with col3:
    if st.session_state.analysis_complete:
        if st.button("Current Analysis", type="secondary", disabled=not st.session_state.show_history):
            st.session_state.show_history = False
            st.experimental_rerun()

with col4:
    if st.button("New Analysis", type="primary"):
        st.session_state.current_analysis = None
        st.session_state.analysis_complete = False
        st.session_state.show_history = False
        st.experimental_rerun()

# Status bar
if st.session_state.analysis_complete and st.session_state.current_analysis:
    analysis = st.session_state.current_analysis
    session_info = analysis['session_info']
    st.info(f"Currently Loaded: {session_info['topic']} | Client: {session_info['client_site']}")
elif st.session_state.show_history:
    st.info("Browsing History - Your current analysis is preserved")

# Show History Section
if st.session_state.show_history:
    st.markdown("---")
    st.header("Analysis History")
    
    try:
        sessions = db.load_analysis_sessions()
        
        if not sessions:
            st.info("No analysis sessions found. Run a new analysis to get started!")
        else:
            search_term = st.text_input("Search by topic or client site:")
            
            # Simple search filter
            if search_term:
                sessions = [s for s in sessions if 
                          search_term.lower() in s['topic'].lower() or 
                          search_term.lower() in s['client_site'].lower()]
            
            for session in sessions:
                current_session_id = None
                if st.session_state.analysis_complete and st.session_state.current_analysis:
                    current_session_id = st.session_state.current_analysis['session_info'].get('session_id')
                
                is_current = (current_session_id == session['id'])
                
                with st.expander(
                    f"{'CURRENT - ' if is_current else ''}{session['timestamp']} - {session['topic']} ({session['client_site']})",
                    expanded=is_current
                ):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**Topic:** {session['topic']}")
                        st.write(f"**Client Site:** {session['client_site']}")
                        st.write(f"**Sites Analyzed:** {session['num_sites']} competitors, max {session['max_pages']} pages each")
                        if is_current:
                            st.success("This is your currently loaded analysis")
                    
                    with col2:
                        if not is_current:
                            if st.button(f"Load This Analysis", key=f"load_{session['id']}"):
                                session_info, competitor_results, client_results = db.load_session_details(session['id'])
                                if session_info:
                                    st.session_state.current_analysis = {
                                        'session_info': session_info,
                                        'competitor_results': competitor_results,
                                        'client_results': client_results
                                    }
                                    st.session_state.analysis_complete = True
                                    st.session_state.show_history = False
                                    st.success(f"Loaded analysis: {session_info['topic']}")
                                    st.experimental_rerun()
                                else:
                                    st.error("Failed to load session data")
                        else:
                            st.button("Currently Loaded", disabled=True, key=f"current_{session['id']}")
                    
                    with col3:
                        if not is_current:
                            if st.button(f"Delete", key=f"del_{session['id']}"):
                                db.delete_session(session['id'])
                                st.success("Session deleted!")
                                st.experimental_rerun()
                        else:
                            st.write("Active")
    
    except Exception as e:
        st.error(f"Database error: {e}")

# Main Analysis Interface
if not st.session_state.analysis_complete and not st.session_state.show_history:
    with st.form("analysis_form"):
        st.subheader("Configure Your SEO Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Enter a topic/keyword:", placeholder="e.g., blockchain consulting")
            num_sites = st.slider("Number of competitor sites", 1, 5, 2)
        with col2:
            client_site = st.text_input("Enter your client website URL:", placeholder="https://example.com")
            max_pages = st.slider("Max pages per site", 1, 20, 5)

        submitted = st.form_submit_button("Run Analysis", type="primary")
        
        if submitted:
            if not topic.strip():
                st.error("Please enter a topic/keyword first.")
            elif not client_site.strip():
                st.error("Please enter a client website URL.")
            elif not client_site.startswith(('http://', 'https://')):
                st.error("Please include http:// or https:// in your client URL.")
            else:
                # Create progress indicators using basic Streamlit components
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                with status_placeholder.container():
                    st.info("🚀 Running SEO Analysis...")
                    
                with progress_placeholder.container():
                    progress_container = st.container()
                    
                    with progress_container:
                        st.write(f"Searching competitors for: **{topic}**")
                        root_links = search_topic(topic, num_sites)
                        
                        if not root_links:
                            st.error("No competitor sites found. Try a different keyword.")
                            st.stop()
                        
                        st.write(f"Found {len(root_links)} competitor sites")
                        for i, link in enumerate(root_links, 1):
                            st.write(f"   {i}. {link}")

                    all_results = []

                    # Competitor crawl
                    for i, root in enumerate(root_links, 1):
                        with progress_container:
                            st.write(f"**Step {i}/{len(root_links)}:** Analyzing {root}")
                        
                        site_results = crawl_site(root, topic, max_pages, progress_container, client=False)
                        all_results.extend(site_results)
                        
                        with progress_container:
                            st.write(f"   Found {len(site_results)} pages")

                    # Client crawl
                    with progress_container:
                        st.write(f"**Final Step:** Analyzing client site: {client_site}")
                    
                    client_results = crawl_site(client_site, topic, max_pages, progress_container, client=True)

                    with progress_container:
                        st.write(f"**Analysis Complete!**")
                        st.write(f"   • Competitor pages scraped: **{len(all_results)}**")
                        st.write(f"   • Client pages scraped: **{len(client_results)}**")

                    if all_results and client_results:
                        # AI Recommendations
                        with progress_container:
                            st.write("Generating AI recommendations...")
                        
                        # Get best competitor content
                        all_results_sorted = sorted(all_results, key=lambda x: x['seo_score'], reverse=True)
                        top_competitors = all_results_sorted[:3]
                        competitor_text = " ".join([comp["content"] for comp in top_competitors])
                        client_text = " ".join([client["content"] for client in client_results])

                        prompt = f"""
                        You are an SEO expert analyzing a website for the keyword "{topic}".

                        COMPETITOR ANALYSIS (Top 3 pages):
                        {competitor_text[:8000]}

                        CLIENT CURRENT CONTENT:
                        {client_text[:6000]}

                        Please provide:
                        1. **SEO Audit Summary** - What's working and what's not
                        2. **Specific Recommendations** - Actionable improvements
                        3. **Optimized Content Rewrite** - Improved title, meta description, H1/H2 structure, and key content sections

                        Focus on beating the competitors while maintaining natural, valuable content.
                        """

                        suggestions = groq_generate(prompt)

                        # Save to database
                        try:
                            session_id = db.save_analysis_session(
                                topic, client_site, num_sites, max_pages, 
                                suggestions, all_results, client_results
                            )
                            
                            with progress_container:
                                st.write(f"Saved to database (Session ID: {session_id})")

                        except Exception as e:
                            st.warning(f"Failed to save to database: {e}")
                            session_id = "unsaved"

                        # Store in session state
                        st.session_state.current_analysis = {
                            'session_info': {
                                'topic': topic,
                                'client_site': client_site,
                                'num_sites': num_sites,
                                'max_pages': max_pages,
                                'ai_recommendations': suggestions,
                                'session_id': session_id,
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            },
                            'competitor_results': all_results,
                            'client_results': client_results
                        }
                        st.session_state.analysis_complete = True
                        
                        # Clear progress indicators and show success
                        progress_placeholder.empty()
                        status_placeholder.success("✅ Analysis completed and saved!")
                        time.sleep(1)  # Brief pause to show success message
                        st.experimental_rerun()
                    else:
                        progress_placeholder.empty()
                        status_placeholder.error("❌ Insufficient data collected. Try increasing the number of sites or pages per site.")

# Display Current Analysis Results
if st.session_state.analysis_complete and st.session_state.current_analysis and not st.session_state.show_history:
    analysis = st.session_state.current_analysis
    session_info = analysis['session_info']
    competitor_results = analysis['competitor_results']
    client_results = analysis['client_results']
    
    st.success("Analysis Results Ready")
    
    # Analysis info header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"**Topic:** {session_info['topic']} | **Client:** {session_info['client_site']}")
    with col2:
        st.metric("Session ID", session_info.get('session_id', 'N/A'))
    
    # Main results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Competitors", "Client Pages", "AI Strategy"])
    
    with tab1:
        st.subheader("Analysis Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Competitor Pages", len(competitor_results))
        with col2:
            st.metric("Client Pages", len(client_results))
        with col3:
            if competitor_results:
                avg_comp_score = sum(r['seo_score'] for r in competitor_results) / len(competitor_results)
                st.metric("Competitor Avg Score", f"{avg_comp_score:.1f}")
            else:
                st.metric("Competitor Avg Score", "0")
        with col4:
            if client_results:
                avg_client_score = sum(r['seo_score'] for r in client_results) / len(client_results)
                st.metric("Client Avg Score", f"{avg_client_score:.1f}")
            else:
                st.metric("Client Avg Score", "0")
    
    with tab2:
        st.subheader("Competitor Analysis")
        
        if competitor_results:
            # Sort by score
            sorted_competitors = sorted(competitor_results, key=lambda x: x['seo_score'], reverse=True)
            best_competitor = sorted_competitors[0]
            st.success(f"**Top Competitor:** {best_competitor['url']} (Score: {best_competitor['seo_score']})")
            
            # Display as table
            st.subheader("All Competitor Pages")
            for i, comp in enumerate(sorted_competitors, 1):
                with st.expander(f"{i}. {comp['url']} (Score: {comp['seo_score']})"):
                    st.write(f"**Title:** {comp['title']}")
                    st.write(f"**Meta Description:** {comp['meta']}")
                    st.write(f"**H1:** {comp['h1']}")
                    st.write(f"**Word Count:** {comp['word_count']}")
        else:
            st.warning("No competitor data available")
    
    with tab3:
        st.subheader("Your Client's Pages")
        
        if client_results:
            sorted_client = sorted(client_results, key=lambda x: x['seo_score'], reverse=True)
            best_client = sorted_client[0]
            st.info(f"**Best Client Page:** {best_client['url']} (Score: {best_client['seo_score']})")
            
            for i, client in enumerate(sorted_client, 1):
                with st.expander(f"{i}. {client['url']} (Score: {client['seo_score']})"):
                    st.write(f"**Title:** {client['title']}")
                    st.write(f"**Meta Description:** {client['meta']}")
                    st.write(f"**H1:** {client['h1']}")
                    st.write(f"**Word Count:** {client['word_count']}")
        else:
            st.warning("No client data available")
    
    with tab4:
        st.subheader("AI-Powered SEO Strategy")
        st.markdown(session_info['ai_recommendations'])
    
    # Download Section
    st.markdown("---")
    st.subheader("Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if competitor_results:
            headers = ['url', 'title', 'meta', 'h1', 'word_count', 'seo_score']
            csv_content = create_csv(competitor_results, headers)
            st.download_button(
                "Competitor Data (CSV)", 
                csv_content.encode("utf-8"), 
                f"competitors_{session_info['topic'].replace(' ', '_')}.csv", 
                "text/csv",
                key="download_comp"
            )
    
    with col2:
        if client_results:
            headers = ['url', 'title', 'meta', 'h1', 'word_count', 'seo_score']
            csv_content = create_csv(client_results, headers)
            st.download_button(
                "Client Data (CSV)", 
                csv_content.encode("utf-8"), 
                f"client_{session_info['topic'].replace(' ', '_')}.csv", 
                "text/csv",
                key="download_client"
            )
    
    with col3:
        st.download_button(
            "AI Strategy (TXT)", 
            session_info['ai_recommendations'].encode("utf-8"), 
            f"seo_strategy_{session_info['topic'].replace(' ', '_')}.txt", 
            "text/plain",
            key="download_ai"
        )

# Sidebar
with st.sidebar:
    st.header("System Status")
    
    if st.session_state.analysis_complete:
        st.success("Analysis Loaded")
        analysis = st.session_state.current_analysis
        if analysis and 'session_info' in analysis:
            st.write(f"**Topic:** {analysis['session_info']['topic']}")
            competitor_count = len(analysis['competitor_results'])
            client_count = len(analysis['client_results'])
            st.write(f"**Pages:** {competitor_count + client_count} total")
    else:
        st.info("Ready for New Analysis")
    
    st.markdown("---")
    st.header("Database Status")
    
    st.write("**Type:** SQLite (Pandas-Free)")
    
    try:
        sessions = db.load_analysis_sessions()
        st.metric("Stored Sessions", len(sessions))
    except Exception:
        st.metric("Stored Sessions", "Error loading")
    
    if st.button("Clear All History"):
        if st.checkbox("Confirm deletion of all analysis data"):
            try:
                if os.path.exists(LOCAL_DB_PATH):
                    os.remove(LOCAL_DB_PATH)
                db.init_database()
                st.success("History cleared!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to clear: {e}")
    
    st.markdown("---")
    st.info("This pandas-free version avoids compilation issues while maintaining full functionality.")
