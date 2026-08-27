from typing import Any

from pydantic import BaseModel, Field


# 1. Graph Canvas Schemas
class GraphNode(BaseModel):
    id: str
    label: str
    type: str  # for, "Suspect", "BankAccount", "Phone"
    risk_score: float = Field(
        ..., ge=0, le=100, description="Risk score between 0 and 100"
    )
    group: str | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str  # for "CALL_LOG", "MONEY_TRANSFER", "OWNS_DEVICE"
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    doc_id: str | None = None


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


# 2. Entity Intelligence Schemas
class RiskBreakdown(BaseModel):
    call_frequency_score: float = Field(..., ge=0, le=100)
    cross_case_score: float = Field(..., ge=0, le=100)
    centrality_score: float = Field(..., ge=0, le=100)
    financial_anomaly_score: float = Field(..., ge=0, le=100)


class EntityDetailResponse(BaseModel):
    id: str
    name: str
    type: str
    overall_risk_score: float = Field(..., ge=0, le=100)
    risk_breakdown: RiskBreakdown
    metadata: dict[str, Any] | None = None


# 3. Legal Evidence Schema (Audit Drawer)
class EvidenceItem(BaseModel):
    doc_id: str
    doc_type: str  # for "FIR", "BankStatement", "CDR"
    excerpt: str
    timestamp: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# 4. Ingestion Schema (NLP Extractor Contract)
class IngestionPayload(BaseModel):
    entities: list[dict[str, Any]]
    relationships: list[dict[str, Any]]


# 5. Agent Query Schemas
class AgentQueryRequest(BaseModel):
    query: str


class AgentQueryResponse(BaseModel):
    query: str
    cypher_generated: str
    answer: str
    highlighted_nodes: list[str]
