import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_sync_playwright_browser,  
)
from dotenv import load_dotenv 
import os
import json

load_dotenv() 

# Load your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY env variable")

# Read CSV with URLs
df = pd.read_csv("social_services.csv")  # CSV must have a 'website_url' column
print("Read CSV")

# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)

# Setup LangChain Browser Toolkit
sync_browser = create_sync_playwright_browser()
browser_toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)

# Create the tools from the toolkit (browse, click, etc.)
tools = browser_toolkit.get_tools()

# Initialize the agent with the browser tools and LLM
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    max_iterations=20  # limit steps to avoid infinite loops
)
print("Initialized agent")

results = []

for url in df['website_url']:
    print(f"\nProcessing: {url}")
    try:
        task_prompt = f"""
        You are an expert web scraping assistant. Your goal is to find specific information by navigating directly to the most relevant pages.

        **Your strategy is to navigate directly to URLs, not to click through menus.**

        1.  Start at the main URL: {url}.
        2.  Use the `get_elements` tool with the selector 'a' to get a JSON list of all available links and their text (e.g., `get_elements(selector='a', attributes=['href', 'innerText'])`).
        3.  Analyze this list. Find the URLs that are **most likely** to contain the information you need. Look for keywords like 'about', 'services', 'programs', or 'contact' in the 'href' or 'innerText'.
        4.  Once you identify the best URLs, use the `Maps_browser` tool to go directly to those pages. For example, if you find a link like `{{'href': '/about-us', 'innerText': 'About Our Mission'}}`, you should run `Maps_browser(url='{url}/about-us')`.
        5.  After navigating, use `extract_text` to get the content of the new page.
        6.  From this text, extract the required fields. Navigate to many (at least 5) different links using steps 3-5 to find as much information as possible. 
        7.  Return **ONLY a single JSON object** with the following exact keys. Do not add any text before or after the JSON. 

        - "name" (organization name)
        - "description" (a 1-2 sentence description of the services the organization provides) 
        - "operational_hours" (days and hours of operation; please specify if hours differ by day; write "None" if none; if truly no information can be found, write "Unknown")
        - "phone_number" (phone number; write all phone numbers if multiple are found) 
        - "email" (email; write all emails if multiple are found) 
        - "address" (address; write all addresses if multiple are found) 
        - "cost" (description of cost of services; write "Free" if none; if truly no information can be found, write "Unknown") 
        - "counties_served" (list of counties served; write cities, states, or regions served if counties aren't available; if truly no information can be found, write "Unknown") 
        - "eligibility_requirements" (description of eligibility requirements; write "None" if none; if truly no information can be found, write "Unknown") 
        - "ada_accessible" ("yes" if services are ADA accessible, "no" if not; write "None" if none; if truly no information can be found, write "Unknown") 
        - "citizenship_requirements" (description of citizenship requirements; write "None" if none; if truly no information can be found, write "Unknown") 

        Do not try to click on menu headers like 'Programs' or 'Services'; instead go directly to the sub-page URL.
        """

        extraction_response = agent.run(task_prompt)

        # Try to parse JSON from agent output
        try:
            data = json.loads(extraction_response)
        except Exception:
            print("Warning: Could not parse JSON, saving raw text.")
            data = {"raw_output": extraction_response}

        data["url"] = url
        results.append(data)

    except Exception as e:
        print(f"Error processing {url}: {e}")
        results.append({"url": url, "error": str(e)})

# Save results to CSV
print(results) 

