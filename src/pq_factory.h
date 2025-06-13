// pq_factory.h
#pragma once
#include "pq_table_base.h"
#include "rabitq_pq_table.h"

enum class PQType { PQ, RaBitQ, LSQ };

template <typename T>
class PQFactory {
  public:
    static std::unique_ptr<PQTableBase<T>> create_pq_table(PQType type) {
        switch(type) {
        case Type::PQ:
            return std::make_unique<FixedChunkPQTableAdapter<T>>();
        case Type::RaBitQ:
            return std::make_unique<RabitQPQTableAdapter<T>>();
        default:
            return nullptr;
        }
    }
};
