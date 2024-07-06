from typing import Optional, Dict, Any
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

class HeartFailureInput(BaseModel):
    current_step: str = Field(description="The current step in the assessment process")
    response: str = Field(description="The doctor's response to the current question")

class EmergencyHeartFailureTool(BaseTool):
    name: str = "Emergency Heart Failure Management"
    description: str = "Guides doctors through rapid assessment and management of heart failure in emergency situations"
    args_schema: type[BaseModel] = HeartFailureInput
    return_direct: bool = True

    assessment_steps: Dict[str, str] = {
        "step1": "What is the patient's ejection fraction (EF)? (Enter as a percentage)",
        "step2": "Are there signs/symptoms of heart failure? (yes/no)",
        "step3": "How severe are the symptoms? (mild/moderate/severe)",
    }
    step_logic: Dict[str, Any] = {
        "step1": lambda x: "step2" if x.isdigit() and 0 <= int(x) <= 100 else "Invalid EF. Please enter a number between 0 and 100.",
        "step2": lambda x: "step3" if x.lower() in ["yes", "no"] else "Invalid response. Please answer 'yes' or 'no'.",
        "step3": lambda x: "classification" if x.lower() in ["mild", "moderate", "severe"] else "Invalid severity. Please enter 'mild', 'moderate', or 'severe'.",
    }

    def __init__(self, name: str = None, description: str = None):
        super().__init__()
        self.name = name or self.name
        self.description = description or self.description

    def _classify_hf(self, ef: int, has_symptoms: bool, severity: str) -> str:
        if ef >= 50:
            return "HFpEF" if has_symptoms else "At risk for HF"
        elif 40 <= ef < 50:
            return "HFmrEF"
        else:
            return "HFrEF"

    def _get_recommendations(self, classification: str, severity: str) -> str:
        recommendations = {
            "At risk for HF": "Monitor closely. Consider preventive measures.",
            "HFpEF": "Optimize blood pressure control. Consider diuretics for congestion.",
            "HFmrEF": "Similar to HFrEF management. Optimize GDMT.",
            "HFrEF": "Initiate or optimize GDMT (ACEi/A70RB/ARNI, beta-blockers, MRAs). Consider device therapy if EF ≤ 35%."
        }
        urgent_care = "Urgent: Provide oxygen if needed. Consider IV diuretics for congestion. Monitor vital signs closely."
        return f"{recommendations[classification]}\n{urgent_care if severity in ['moderate', 'severe'] else ''}"

    def _run(
        self,
        current_step: str,
        response: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Process the current step and response in the HF assessment."""
        if current_step not in self.assessment_steps and current_step != "classification":
            return "Invalid step. Please start with 'step1'."
        
        if current_step == "classification":
            ef, has_symptoms, severity = response.split(',')
            classification = self._classify_hf(int(ef), has_symptoms.lower() == 'yes', severity.lower())
            recommendations = self._get_recommendations(classification, severity.lower())
            return f"Classification: {classification}\n\nRecommendations:\n{recommendations}"
        
        next_step_or_message = self.step_logic[current_step](response)
        
        if next_step_or_message in self.assessment_steps:
            return f"{next_step_or_message}|{self.assessment_steps[next_step_or_message]}"
        elif next_step_or_message == "classification":
            return f"classification|Please confirm the following:\nEF: {self.ef}, Symptoms: {self.has_symptoms}, Severity: {self.severity}"
        else:
            return next_step_or_message

    async def _arun(
        self,
        current_step: str,
        response: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Process the current step and response asynchronously."""
        return self._run(current_step, response, run_manager)
