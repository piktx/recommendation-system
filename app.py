import streamlit as st
import nest_asyncio
from llama_index.core import Settings
from llama_index.llms.sambanovasystems import SambaNovaCloud
from duckduckgo_search import DDGS
from llama_index.core.tools import FunctionTool
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner

# Apply nest_asyncio
nest_asyncio.apply()

# Full Product Categories Configuration
PRODUCT_CATEGORIES = {
    "Cameras": {
        "features": [
            "Low Light Performance", "4K Video", "Image Stabilization",
            "Weather Sealing", "Compact Size", "WiFi Connectivity", "Touch Screen"
        ],
        "types": [
            "Mirrorless", "DSLR", "Point and Shoot", "Medium Format"
        ],
        "use_cases": [
            "Professional Photography", "Vlogging", "Travel Photography",
            "Sports Photography", "Wildlife Photography"
        ]
    },
    "Laptops": {
        "features": [
            "Long Battery Life", "Dedicated Graphics", "Touch Screen",
            "Backlit Keyboard", "Fingerprint Reader", "Thunderbolt Ports", "5G Connectivity"
        ],
        "types": [
            "Ultrabook", "Gaming Laptop", "Business Laptop",
            "2-in-1 Convertible", "Budget Laptop"
        ],
        "use_cases": [
            "Gaming", "Content Creation", "Business", "Student", "Programming"
        ]
    },
    "Smartphones": {
        "features": [
            "5G Support", "Wireless Charging", "Water Resistance",
            "Face Recognition", "Multiple Cameras", "Fast Charging", "NFC"
        ],
        "types": [
            "Flagship", "Mid-range", "Budget", "Gaming Phone", "Compact"
        ],
        "use_cases": [
            "Photography", "Gaming", "Business", "Basic Use", "Content Creation"
        ]
    },
    "Smart Home Devices": {
        "features": [
            "Voice Control", "Mobile App Control", "Energy Monitoring",
            "Motion Detection", "Smart Scheduling", "Multi-user Support", "Integration Capabilities"
        ],
        "types": [
            "Smart Speakers", "Security Cameras", "Smart Lights",
            "Smart Thermostats", "Smart Displays"
        ],
        "use_cases": [
            "Home Security", "Energy Management", "Entertainment",
            "Home Automation", "Family Organization"
        ]
    }
}

def initialize_llm(api_key: str):
    """Initialize the SambaNova LLM with specific parameters"""
    if not api_key:
        raise ValueError("API key is required")
        
    return SambaNovaCloud(
        model="Meta-Llama-3.1-70B-Instruct",
        sambaverse_api_key=api_key,
        context_window=10000,
        max_tokens=2048,
        temperature=0.1,
        top_k=1,
        top_p=0.95,
        additional_kwargs={
            "return_raw": True,
            "format_response": False
        }
    )

def search(query: str) -> str:
    """Perform DuckDuckGo search"""
    try:
        req = DDGS()
        response = req.text(query, max_results=4)
        context = ""
        for result in response:
            context += result['body']
        return context
    except Exception as e:
        return f"Search failed: {str(e)}"

def setup_agent(api_key: str):
    """Setup the LATS agent with search tool"""
    if not api_key:
        st.error("API key is required")
        return None
        
    try:
        llm = initialize_llm(api_key)
        Settings.llm = llm
        
        search_tool = FunctionTool.from_defaults(
            fn=search,
            name="search",
            description="Search for product information and reviews"
        )
        
        agent_worker = LATSAgentWorker(
            tools=[search_tool],
            num_expansions=2,
            max_rollouts=2,
            verbose=True,
            llm=llm
        )
        
        return AgentRunner(agent_worker)
    except Exception as e:
        st.error(f"Agent setup failed: {str(e)}")
        return None

def process_recommendation(query: str, agent: AgentRunner):
    """Process the recommendation query with error handling"""
    try:
        response = agent.chat(query).response
        if "I am still thinking." in response:
            return agent.list_tasks()[-1].extra_state["root_node"].children[0].children[0].current_reasoning[-1].observation
        else:
            return response
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}"

def main():
    st.set_page_config(page_title="Smart Product Recommendation System", layout="wide")
    
    # Title and description
    st.title("🎯 Smart Product Recommendation System")
    st.write("""
    Get personalized product recommendations based on your requirements. 
    Our AI-powered system analyzes current market offerings to find the best match for your needs.
    """)
    
    # Initialize session state for agent
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Authentication")
        api_key = st.text_input("Enter SambaNova API Key:", 
                              type="password",
                              help="Get your API key from SambaNova Cloud portal",
                              value="")
        
        if api_key and not st.session_state.agent:
            with st.spinner("Initializing AI engine..."):
                st.session_state.agent = setup_agent(api_key)
    
    st.header("What are you looking for?")
    
    # Rest of the UI components (unchanged)
    # ... (keep existing category selection, input parameters, and recommendation logic)

if __name__ == "__main__":
    main()
