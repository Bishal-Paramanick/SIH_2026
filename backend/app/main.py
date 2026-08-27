from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    EntityDetailResponse,
    EvidenceItem,
    GraphResponse,
    IngestionPayload,
)

app = FastAPI(
    title="Crime Network Graph Intelligence API",
    description="Backend services for crime pattern detection, evidence auditing, and graph visualization.",
    version="1.0.0",
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 1. Graph Canvas Endpoint ---
@app.get("/api/graph", response_model=GraphResponse, tags=["Graph Canvas"])
def get_graph():
    return {
        "nodes": [
            {
                "id": "suspect_101",
                "label": "Vikram Singh",
                "type": "Suspect",
                "risk_score": 88.5,
                "group": "Syndicate A",
            },
            {
                "id": "suspect_102",
                "label": "Rahul Sharma",
                "type": "Suspect",
                "risk_score": 64.0,
                "group": "Syndicate A",
            },
            {
                "id": "bank_9901",
                "label": "HDFC Acc ...4421",
                "type": "BankAccount",
                "risk_score": 92.0,
                "group": "Mule Accounts",
            },
            {
                "id": "phone_8812",
                "label": "+91-9876543210",
                "type": "Phone",
                "risk_score": 45.0,
                "group": "Communications",
            },
        ],
        "edges": [
            {
                "source": "suspect_101",
                "target": "phone_8812",
                "type": "OWNS_DEVICE",
                "confidence": 0.95,
                "doc_id": "FIR-2026-99",
            },
            {
                "source": "suspect_101",
                "target": "bank_9901",
                "type": "MONEY_TRANSFER",
                "confidence": 0.87,
                "doc_id": "STMT-2026-03",
            },
            {
                "source": "suspect_102",
                "target": "bank_9901",
                "type": "MONEY_TRANSFER",
                "confidence": 0.91,
                "doc_id": "STMT-2026-04",
            },
        ],
    }


# --- 2. Entity Intelligence Detail Endpoint ---
@app.get(
    "/api/entity/{id}",
    response_model=EntityDetailResponse,
    tags=["Entity Intelligence"],
)
def get_entity_detail(id: str):
    return {
        "id": id,
        "name": "Vikram Singh",
        "type": "Suspect",
        "overall_risk_score": 88.5,
        "risk_breakdown": {
            "call_frequency_score": 75.0,
            "cross_case_score": 90.0,
            "centrality_score": 85.5,
            "financial_anomaly_score": 95.0,
        },
        "metadata": {
            "primary_alias": "Vicky",
            "active_jurisdictions": ["Kolkata", "Delhi"],
        },
    }


# --- 3. Legal Evidence Audit Drawer Endpoint ---
@app.get(
    "/api/entity/{id}/evidence",
    response_model=list[EvidenceItem],
    tags=["Legal Evidence"],
)
def get_entity_evidence(id: str):
    return [
        {
            "doc_id": "FIR-2026-991A",
            "doc_type": "FIR",
            "excerpt": "Suspect intercepted coordinating hawala funds transfer near Park Street corridor.",
            "timestamp": "2026-08-15T14:30:00Z",
            "confidence": 0.94,
        },
        {
            "doc_id": "CDR-2026-0044",
            "doc_type": "CDR",
            "excerpt": "Frequent outbound satellite call bursts detected prior to transaction window.",
            "timestamp": "2026-08-16T22:15:00Z",
            "confidence": 0.89,
        },
    ]


# --- 4. Ingestion Endpoint (NLP Extractor Contract) ---
@app.post("/api/ingest", status_code=status.HTTP_200_OK, tags=["Ingestion"])
def ingest_data(payload: IngestionPayload):
    return {
        "status": "success",
        "message": f"Successfully ingested {len(payload.entities)} entities and {len(payload.relationships)} relationships.",
    }


# --- 5. Agent Natural Language Query Endpoint ---
@app.post(
    "/api/agent/query",
    response_model=AgentQueryResponse,
    tags=["Agent Copilot"],
)
def query_agent(request: AgentQueryRequest):
    return {
        "query": request.query,
        "cypher_generated": "MATCH (s:Suspect)-[r:MONEY_TRANSFER]->(b:BankAccount) WHERE b.risk_score > 80 RETURN s, r, b",
        "answer": f"Identified high-risk laundering pipeline connecting Vikram Singh to mule account HDFC ...4421 based on query: '{request.query}'.",
        "highlighted_nodes": ["suspect_101", "bank_9901"],
    }
