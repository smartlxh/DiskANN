// pq_factory.h
#pragma once
#include "pq_table_base.h"

enum class PQType { PQ, OPQ, RaBitQ, LSQ };

template <typename T>
class PQFactory {
  public:
    static std::unique_ptr<PQTableBase<T>> create_pq_table(PQType type) {
        switch(type) {
        case Type::PQ:
        case Type::OPQ:
            return std::make_unique<FixedChunkPQTableAdapter<T>>();
        default:
            return nullptr;
        }
    }
};
