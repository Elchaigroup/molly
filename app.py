import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import pandas as pd
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
    st.error("🔑 Missing API Keys! Please configure SCRAPINGDOG_API_KEY and GROQ_API_KEY in environment variables.")
    st.info("📝 Contact your administrator to set up the required API keys.")
    st.stop()

# -------------------
# DATABASE CONFIGURATION (Future-ready for online DB)
# -------------------
# Database settings - easily switchable to PostgreSQL later
USE_ONLINE_DB = os.getenv("USE_ONLINE_DB", "false").lower() == "true"
DATABASE_URL = os.getenv("DATABASE_URL")  # For future PostgreSQL
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
            # PostgreSQL syntax
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
            # SQLite syntax
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
        
        # Insert main session
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
        for _, row in competitor_data.iterrows():
            if self.db_type == "postgresql":
                cursor.execute('''
                    INSERT INTO competitor_results 
                    (session_id, url, title, meta, h1, word_count, seo_score, content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (session_id, row['url'], row['title'], row['meta'], 
                      row['h1'], row['word_count'], row['seo_score'], row['content']))
            else:
                cursor.execute('''
                    INSERT INTO competitor_results 
                    (session_id, url, title, meta, h1, word_count, seo_score, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, row['url'], row['title'], row['meta'], 
                      row['h1'], row['word_count'], row['seo_score'], row['content']))
        
        # Insert client results
        for _, row in client_data.iterrows():
            if self.db_type == "postgresql":
                cursor.execute('''
                    INSERT INTO client_results 
                    (session_id, url, title, meta, h1, word_count, seo_score, content)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ''', (session_id, row['url'], row['title'], row['meta'], 
                      row['h1'], row['word_count'], row['seo_score'], row['content']))
            else:
                cursor.execute('''
                    INSERT INTO client_results 
                    (session_id, url, title, meta, h1, word_count, seo_score, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, row['url'], row['title'], row['meta'], 
                      row['h1'], row['word_count'], row['seo_score'], row['content']))
        
        conn.commit()
        conn.close()
        return session_id
    
    def load_analysis_sessions(self):
        """Load all analysis sessions from database"""
        conn = self.get_connection()
        query = '''
            SELECT id, timestamp, topic, client_site, num_sites, max_pages 
            FROM analysis_sessions 
            ORDER BY timestamp DESC
        '''
        sessions = pd.read_sql_query(query, conn)
        conn.close()
        return sessions
    
    def load_session_details(self, session_id):
        """Load detailed results for a specific session"""
        conn = self.get_connection()
        
        # Load session info
        session_query = '''
            SELECT * FROM analysis_sessions WHERE id = %s
        ''' if self.db_type == "postgresql" else '''
            SELECT * FROM analysis_sessions WHERE id = ?
        '''
        session_info = pd.read_sql_query(session_query, conn, params=(session_id,))
        
        # Load competitor results
        competitor_query = '''
            SELECT * FROM competitor_results WHERE session_id = %s
        ''' if self.db_type == "postgresql" else '''
            SELECT * FROM competitor_results WHERE session_id = ?
        '''
        competitor_results = pd.read_sql_query(competitor_query, conn, params=(session_id,))
        
        # Load client results
        client_query = '''
            SELECT * FROM client_results WHERE session_id = %s
        ''' if self.db_type == "postgresql" else '''
            SELECT * FROM client_results WHERE session_id = ?
        '''
        client_results = pd.read_sql_query(client_query, conn, params=(session_id,))
        
        conn.close()
        
        return session_info.iloc[0] if not session_info.empty else None, competitor_results, client_results
    
    def delete_session(self, session_id):
        """Delete a session and all its related data"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if self.db_type == "postgresql":
            cursor.execute('DELETE FROM competitor_results WHERE session_id = %s', (session_id,))
            cursor.execute('DELETE FROM client_results WHERE session_id = %s', (session_id,))
            cursor.execute('DELETE FROM analysis_sessions WHERE id = %s', (session_id,))
        else:
            cursor.execute('DELETE FROM competitor_results WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM client_results WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM analysis_sessions WHERE id = ?', (session_id,))
        
        conn.commit()
        conn.close()
    
    def get_database_info(self):
        """Get database statistics"""
        if self.db_type == "postgresql":
            return {"type": "PostgreSQL", "persistent": True}
        else:
            size = 0
            if os.path.exists(LOCAL_DB_PATH):
                size = os.path.getsize(LOCAL_DB_PATH) / (1024 * 1024)  # MB
            return {"type": "SQLite", "size_mb": size, "persistent": False}

# Initialize database manager
@st.cache_resource
def get_db_manager():
    return DatabaseManager()

db = get_db_manager()

# -------------------
# ORIGINAL HELPER FUNCTIONS (with better error handling)
# -------------------
def search_topic(topic, num_results=3):
    """Search for competitor sites using ScrapingDog Google API"""
    url = f"https://api.scrapingdog.com/google?api_key={SCRAPINGDOG_API_KEY}&query={topic}&num={num_results}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            st.warning(f"❌ Google search API failed with status {resp.status_code}")
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
        resp = requests.get(scrape_url, timeout=60)  # Longer timeout for scraping
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception:
        return None

def fetch_html_client(url):
    """Direct requests for client site"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=15)
        return resp.text if resp.status_code == 200 else None
    except Exception as e:
        st.warning(f"❌ Client fetch failed: {e}")
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

    # SEO scoring algorithm
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
        "content": text[:2000]  # Limit for storage efficiency
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

        msg = f"🌍 Crawling: {url}"
        if progress_area:
            progress_area.write(msg)

        # Choose appropriate fetch method
        html = fetch_html_client(url) if client else fetch_html(url)
        if not html:
            if progress_area:
                progress_area.write(f"❌ Failed to fetch {url}")
            continue

        results.append(analyze_page(html, url, keyword))

        # Find more internal links to crawl
        for link in extract_links(html, start_url):
            if link not in visited and link not in queue:
                queue.append(link)

        # Rate limiting
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
            return f"⚠️ Groq API Error: {resp.text}"
    except Exception as e:
        return f"⚠️ Groq API Error: {e}"

# -------------------
# SESSION STATE INITIALIZATION
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
st.set_page_config(page_title="SEO Analyzer - Production Ready", layout="wide")

# Header with navigation buttons
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.title("🚀 SEO Analyzer Pro")
    # Show database status
    db_info = db.get_database_info()
    if db_info["type"] == "PostgreSQL":
        st.caption("🗄️ PostgreSQL Database (Persistent)")
    else:
        st.caption("🗄️ SQLite Database")

with col2:
    if st.session_state.show_history:
        if st.button("🔙 Back to Analysis", type="secondary"):
            st.session_state.show_history = False
            st.rerun()
    else:
        if st.button("📚 View History", type="secondary"):
            st.session_state.show_history = True
            st.rerun()

with col3:
    if st.session_state.analysis_complete:
        if st.button("📊 Current Analysis", type="secondary", disabled=not st.session_state.show_history):
            st.session_state.show_history = False
            st.rerun()

with col4:
    if st.button("🆕 New Analysis", type="primary"):
        st.session_state.current_analysis = None
        st.session_state.analysis_complete = False
        st.session_state.show_history = False
        st.rerun()

# Status bar
if st.session_state.analysis_complete and st.session_state.current_analysis:
    analysis = st.session_state.current_analysis
    session_info = analysis['session_info']
    st.info(f"📊 **Currently Loaded:** {session_info['topic']} | Client: {session_info['client_site']} | Session ID: {session_info.get('session_id', 'N/A')}")
elif st.session_state.show_history:
    st.info("📚 **Browsing History** - Your current analysis is preserved")

# Database status warning for SQLite on Render
if db.db_type == "sqlite" and os.getenv("RENDER"):
    st.warning("⚠️ **Render Deployment Note:** SQLite data may be lost on app restart. Upgrade to PostgreSQL for persistence.")

# Show History Section
if st.session_state.show_history:
    st.markdown("---")
    st.header("📚 Analysis History")
    
    try:
        sessions = db.load_analysis_sessions()
        
        if sessions.empty:
            st.info("No analysis sessions found. Run a new analysis to get started!")
        else:
            # Search functionality
            search_term = st.text_input("🔍 Search by topic or client site:")
            if search_term:
                mask = (sessions['topic'].str.contains(search_term, case=False, na=False) | 
                       sessions['client_site'].str.contains(search_term, case=False, na=False))
                sessions = sessions[mask]
            
            # Display sessions
            for idx, session in sessions.iterrows():
                current_session_id = None
                if st.session_state.analysis_complete and st.session_state.current_analysis:
                    current_session_id = st.session_state.current_analysis['session_info'].get('session_id')
                
                is_current = (current_session_id == session['id'])
                
                with st.expander(
                    f"{'🟢 CURRENT - ' if is_current else ''}📅 {session['timestamp']} - {session['topic']} ({session['client_site']})",
                    expanded=is_current
                ):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"**Topic:** {session['topic']}")
                        st.write(f"**Client Site:** {session['client_site']}")
                        st.write(f"**Sites Analyzed:** {session['num_sites']} competitors, max {session['max_pages']} pages each")
                        if is_current:
                            st.success("✅ This is your currently loaded analysis")
                    
                    with col2:
                        if not is_current:
                            if st.button(f"📋 Load This Analysis", key=f"load_{session['id']}"):
                                session_info, competitor_results, client_results = db.load_session_details(session['id'])
                                if session_info is not None:
                                    st.session_state.current_analysis = {
                                        'session_info': session_info,
                                        'competitor_results': competitor_results,
                                        'client_results': client_results
                                    }
                                    st.session_state.analysis_complete = True
                                    st.session_state.show_history = False
                                    st.success(f"✅ Loaded analysis: {session_info['topic']}")
                                    st.rerun()
                                else:
                                    st.error("Failed to load session data")
                        else:
                            st.button("✅ Currently Loaded", disabled=True, key=f"current_{session['id']}")
                    
                    with col3:
                        if not is_current:
                            if st.button(f"🗑️ Delete", key=f"del_{session['id']}"):
                                db.delete_session(session['id'])
                                st.success("Session deleted!")
                                st.rerun()
                        else:
                            st.write("🛡️ Active")
    
    except Exception as e:
        st.error(f"Database error: {e}")
        st.info("Try refreshing the page or contact support.")
    
    st.markdown("---")
    st.info("💡 **Tip:** Your current analysis remains loaded while browsing history.")

# Main Analysis Interface
if not st.session_state.analysis_complete and not st.session_state.show_history:
    with st.form("analysis_form"):
        st.subheader("🔍 Configure Your SEO Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Enter a topic/keyword:", placeholder="e.g., blockchain consulting")
            num_sites = st.slider("Number of competitor sites", 1, 5, 2)
        with col2:
            client_site = st.text_input("Enter your client website URL:", placeholder="https://example.com")
            max_pages = st.slider("Max pages per site", 1, 20, 5)

        # Show estimated time
        estimated_time = (num_sites + 1) * max_pages * 2  # rough estimate in seconds
        st.info(f"⏱️ Estimated analysis time: ~{estimated_time//60}m {estimated_time%60}s")

        submitted = st.form_submit_button("🔍 Run Analysis", type="primary")
        
        if submitted:
            if not topic.strip():
                st.error("Please enter a topic/keyword first.")
            elif not client_site.strip():
                st.error("Please enter a client website URL.")
            elif not client_site.startswith(('http://', 'https://')):
                st.error("Please include http:// or https:// in your client URL.")
            else:
                # Run the analysis
                with st.status("Running SEO Analysis...", expanded=True) as status:
                    progress_container = st.container()
                    
                    with progress_container:
                        st.write(f"🔎 Searching competitors for: **{topic}**")
                        root_links = search_topic(topic, num_sites)
                        
                        if not root_links:
                            st.error("No competitor sites found. Try a different keyword.")
                            st.stop()
                        
                        st.write(f"✅ Found {len(root_links)} competitor sites")
                        for i, link in enumerate(root_links, 1):
                            st.write(f"   {i}. {link}")

                    all_results = []

                    # Competitor crawl
                    for i, root in enumerate(root_links, 1):
                        with progress_container:
                            st.write(f"🔎 **Step {i}/{len(root_links)}:** Analyzing {root}")
                        
                        site_results = crawl_site(root, topic, max_pages, progress_container, client=False)
                        all_results.extend(site_results)
                        
                        with progress_container:
                            st.write(f"   ✅ Found {len(site_results)} pages")

                    df_comp = pd.DataFrame(all_results)

                    # Client crawl
                    with progress_container:
                        st.write(f"📝 **Final Step:** Analyzing client site: {client_site}")
                    
                    client_results = crawl_site(client_site, topic, max_pages, progress_container, client=True)
                    df_client = pd.DataFrame(client_results)

                    # Results summary
                    with progress_container:
                        st.write(f"✅ **Analysis Complete!**")
                        st.write(f"   • Competitor pages scraped: **{len(df_comp)}**")
                        st.write(f"   • Client pages scraped: **{len(df_client)}**")

                    if not df_comp.empty and not df_client.empty:
                        # AI Recommendations
                        with progress_container:
                            st.write("🤖 Generating AI recommendations...")
                        
                        # Get best competitor content
                        top_competitors = df_comp.sort_values("seo_score", ascending=False).head(3)
                        competitor_text = " ".join(top_competitors["content"])
                        client_text = " ".join(df_client["content"])

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
                                suggestions, df_comp, df_client
                            )
                            
                            with progress_container:
                                st.write(f"💾 Saved to database (Session ID: {session_id})")

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
                            'competitor_results': df_comp,
                            'client_results': df_client
                        }
                        st.session_state.analysis_complete = True
                        
                        status.update(label="✅ Analysis completed and saved!", state="complete")
                        st.balloons()  # Celebration effect
                        st.rerun()
                    else:
                        st.error("⚠️ Insufficient data collected. Try:")
                        st.write("• Increasing the number of sites or pages per site")
                        st.write("• Using a more specific keyword")
                        st.write("• Checking if the client site is accessible")

# Display Current Analysis Results
if st.session_state.analysis_complete and st.session_state.current_analysis and not st.session_state.show_history:
    analysis = st.session_state.current_analysis
    session_info = analysis['session_info']
    df_comp = analysis['competitor_results']
    df_client = analysis['client_results']
    
    st.success("✅ Analysis Results Ready")
    
    # Analysis info header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info(f"**🎯 Topic:** {session_info['topic']} | **🏠 Client:** {session_info['client_site']}")
    with col2:
        st.metric("Session ID", session_info.get('session_id', 'N/A'))
    
    # Main results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🏆 Competitors", "📝 Client Pages", "🤖 AI Strategy"])
    
    with tab1:
        st.subheader("📊 Analysis Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Competitor Pages", len(df_comp))
        with col2:
            st.metric("Client Pages", len(df_client))
        with col3:
            avg_comp_score = df_comp['seo_score'].mean() if not df_comp.empty else 0
            st.metric("Competitor Avg Score", f"{avg_comp_score:.1f}")
        with col4:
            avg_client_score = df_client['seo_score'].mean() if not df_client.empty else 0
            delta = avg_client_score - avg_comp_score
            st.metric("Client Avg Score", f"{avg_client_score:.1f}", f"{delta:+.1f}")
        
        # Score comparison chart
        if not df_comp.empty and not df_client.empty:
            st.subheader("📈 SEO Score Comparison")
            
            # Create comparison data
            comp_scores = df_comp['seo_score'].tolist()
            client_scores = df_client['seo_score'].tolist()
            
            # Simple bar chart using Streamlit
            chart_data = pd.DataFrame({
                'Competitor Pages': comp_scores + [0] * (max(len(client_scores) - len(comp_scores), 0)),
                'Client Pages': [0] * (max(len(comp_scores) - len(client_scores), 0)) + client_scores
            })
            
            st.bar_chart(chart_data)
    
    with tab2:
        st.subheader("🏆 Competitor Analysis")
        
        # Top performer
        if not df_comp.empty:
            best_competitor = df_comp.sort_values("seo_score", ascending=False).iloc[0]
            st.success(f"🏆 **Top Competitor:** {best_competitor['url']} (Score: {best_competitor['seo_score']})")
            
            with st.expander("🔍 View Top Competitor Details"):
                st.write(f"**Title:** {best_competitor['title']}")
                st.write(f"**Meta Description:** {best_competitor['meta']}")
                st.write(f"**H1:** {best_competitor['h1']}")
                st.write(f"**Word Count:** {best_competitor['word_count']}")
        
        # All competitors table
        st.subheader("📊 All Competitor Pages")
        if not df_comp.empty:
            display_cols = ["url", "title", "seo_score", "word_count"]
            st.dataframe(
                df_comp[display_cols].sort_values("seo_score", ascending=False), 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No competitor data available")
    
    with tab3:
        st.subheader("📝 Your Client's Pages")
        
        if not df_client.empty:
            # Best client page
            best_client = df_client.sort_values("seo_score", ascending=False).iloc[0]
            st.info(f"⭐ **Best Client Page:** {best_client['url']} (Score: {best_client['seo_score']})")
            
            # All client pages
            display_cols = ["url", "title", "seo_score", "word_count"]
            st.dataframe(
                df_client[display_cols].sort_values("seo_score", ascending=False), 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No client data available")
    
    with tab4:
        st.subheader("🤖 AI-Powered SEO Strategy")
        
        # Display recommendations in a nice format
        recommendations = session_info['ai_recommendations']
        
        if "⚠️" in recommendations:
            st.error("AI recommendations failed to generate properly. Please try running the analysis again.")
        else:
            st.markdown(recommendations)
        
        # Action items extraction
        st.markdown("---")
        st.subheader("✅ Quick Action Items")
        st.info("💡 **Pro Tip:** Use the AI recommendations above to implement these improvements on your client's website.")
    
    # Download Section
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not df_comp.empty:
            csv_comp = df_comp.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Competitor Data (CSV)", 
                csv_comp, 
                f"competitors_{session_info['topic'].replace(' ', '_')}_{session_info.get('session_id', 'current')}.csv", 
                "text/csv",
                key="download_comp"
            )
    
    with col2:
        if not df_client.empty:
            csv_client = df_client.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Client Data (CSV)", 
                csv_client, 
                f"client_{session_info['topic'].replace(' ', '_')}_{session_info.get('session_id', 'current')}.csv", 
                "text/csv",
                key="download_client"
            )
    
    with col3:
        st.download_button(
            "📥 AI Strategy (TXT)", 
            session_info['ai_recommendations'].encode("utf-8"), 
            f"seo_strategy_{session_info['topic'].replace(' ', '_')}_{session_info.get('session_id', 'current')}.txt", 
            "text/plain",
            key="download_ai"
        )

# Sidebar with enhanced info
with st.sidebar:
    st.header("🎛️ System Status")
    
    # Current analysis status
    if st.session_state.analysis_complete:
        st.success("✅ Analysis Loaded")
        analysis = st.session_state.current_analysis
        if analysis and 'session_info' in analysis:
            st.write(f"**📊 Topic:** {analysis['session_info']['topic']}")
            competitor_count = len(analysis['competitor_results'])
            client_count = len(analysis['client_results'])
            st.write(f"**📄 Pages:** {competitor_count + client_count} total")
    else:
        st.info("⏳ Ready for New Analysis")
    
    st.markdown("---")
    st.header("🗄️ Database Status")
    
    db_info = db.get_database_info()
    
    if db_info["type"] == "SQLite":
        st.write(f"**Type:** {db_info['type']}")
        if "size_mb" in db_info:
            st.metric("Database Size", f"{db_info['size_mb']:.2f} MB")
        
        # Session count
        try:
            sessions = db.load_analysis_sessions()
            st.metric("Stored Sessions", len(sessions))
        except Exception:
            st.metric("Stored Sessions", "Error loading")
        
        # Clear history option
        if st.button("🗑️ Clear All History"):
            if st.checkbox("⚠️ Confirm deletion of all analysis data"):
                try:
                    if os.path.exists(LOCAL_DB_PATH):
                        os.remove(LOCAL_DB_PATH)
                    # Reinitialize database
                    db.init_database()
                    st.success("✅ History cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear: {e}")
    
    else:  # PostgreSQL
        st.success(f"**Type:** {db_info['type']} ✅")
        st.write("**Persistence:** Full")
        try:
            sessions = db.load_analysis_sessions()
            st.metric("Stored Sessions", len(sessions))
        except Exception:
            st.metric("Stored Sessions", "Connection Error")
    
    st.markdown("---")
    st.header("🚀 Migration Ready")
    st.info("💾 **Future-Proof:** This app is ready to migrate from SQLite to PostgreSQL by simply setting environment variables!")
    
    if not USE_ONLINE_DB:
        st.write("**Next Steps:**")
        st.write("1. Add PostgreSQL to Render")
        st.write("2. Set `USE_ONLINE_DB=true`")
        st.write("3. Data will automatically migrate!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>🚀 <strong>SEO Analyzer Pro</strong> | Built for Render Cloud Deployment</p>
        <p>💡 Ready for SQLite → PostgreSQL migration</p>
    </div>
    """, 
    unsafe_allow_html=True
)