// pq_factory.h
#pragma once
#include "pq_table_base.h"
#include "fixed_chunk_pq_table_adapter.h"

enum class PQType { PQ, OPQ, RaBitQ, LSQ };

class PQFactory {
  public:
    static std::unique_ptr<PQTableBase<T>> create_pq_table(PQType type) {
        switch(type) {
        case PQType::PQ:
        case PQType::OPQ:
            return std::make_unique<FixedChunkPQTableAdapter>();
        default:
            return nullptr;
        }
    }
};
