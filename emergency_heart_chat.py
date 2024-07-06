import os
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.conversation.base import ConversationChain

from emergency_heart_tools import EmergencyHeartFailureTool

# # Set up environment variables (make sure to set these in your environment)
# GOOGLE_API_KEY = "AIzaSyAlbQppmzEAFV7V_NjoK2PBA44UzBAj9fw"

# # Initialize the Google API model
# google_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
# llm = GoogleGenerativeAI(
#         model="gemini-pro", 
#         google_api_key= "AIzaSyAlbQppmzEAFV7V_NjoK2PBA44UzBAj9fw",
#         temperature=.1,
#         safety_settings={
#             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
#             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
#         }
#     )
# Create the Emergency Heart Failure tool
api_key =""
llm = ChatAnthropic(
    anthropic_api_key=api_key,
    model="claude-3-sonnet-20240320",  # Update this to the latest available model
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2
)
hf_tool = EmergencyHeartFailureTool()

# Create a list of tools for the agent
tools = [
    Tool(
        name="Emergency Heart Failure Management",
        func=hf_tool._run,
        description="Guides through rapid assessment and management of heart failure in emergency situations"
    )
]

# Set up the conversational memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

prompt_template = PromptTemplate(
        input_variables=["history", "human_input"],
        template="""
        You are claude an AI assistant designed to help emergency room doctors quickly assess and manage heart failure patients.
        Use the Emergency Heart Failure Management tool from to guide the doctor through the assessment process.
        Start with 'step1' and use the tool to process each response and determine the next question or action.
        When you receive a response from the tool, if it contains a '|' character, split the response and use the second part as the next question.
        Continue this process until you receive the final classification and recommendations.
        Present the classification and recommendations clearly to the doctor.
        After the assessment, ask if the doctor needs any clarification or has any questions about the recommendations.


        Conversation history:
        {history}

        Doctor: {input}
        AI Assistant: Let me think about that...
        """
    )

# Create the conversational chain

conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt_template,
        input_key="input",  # Align input key
        verbose=True
    )

# Chat loop
print("Welcome to the Emergency Heart Failure Management Assistant!")
print("I'll guide you through a rapid assessment of your patient.")
print("Type 'quit' at any time to exit the chat.")

current_step = "step1"
while True:
    if current_step == "step1":
        print("\nLet's begin the assessment.")
    
    user_input = input("Doctor: ")
    if user_input.lower() == 'quit':
        print("Thank you for using the Emergency Heart Failure Management Assistant. Goodbye!")
        break
    
    response = conversation.predict(input=f"Current step: {current_step}, Doctor's response: {user_input}")
    print(f"Assistant: {response}")

    # Check if we've reached the classification step
    if "Classification:" in response:
        print("\nAssessment complete. Would you like to start another assessment? (Type 'yes' to restart or 'quit' to exit)")
        restart = input("Doctor: ").lower()
        if restart == 'yes':
            current_step = "step1"
            continue
        else:
            print("Thank you for using the Emergency Heart Failure Management Assistant. Goodbye!")
            break

    # Extract the next step from the response
    if "|" in response:
        current_step = response.split("|")[0].strip()
