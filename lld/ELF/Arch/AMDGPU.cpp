//===- AMDGPU.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "Relocations.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Endian.h"

// causes windows build ambiguity
//using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;
using namespace lld;
using namespace lld::elf;

namespace {
class AMDGPU final : public TargetInfo {
private:
  uint32_t calcEFlagsV3() const;
  uint32_t calcEFlagsV4() const;
  uint32_t calcEFlagsV6() const;

public:
  AMDGPU(Ctx &);
  uint32_t calcEFlags() const override;
  void relocate(uint8_t *loc, const Relocation &rel,
                uint64_t val) const override;
  RelExpr getRelExpr(RelType type, const Symbol &s,
                     const uint8_t *loc) const override;
  RelType getDynRel(RelType type) const override;
  int64_t getImplicitAddend(const uint8_t *buf, RelType type) const override;
  void postRelocatePass() const override;
};
} // namespace

AMDGPU::AMDGPU(Ctx &ctx) : TargetInfo(ctx) {
  relativeRel = R_AMDGPU_RELATIVE64;
  gotRel = R_AMDGPU_ABS64;
  symbolicRel = R_AMDGPU_ABS64;
}

static uint32_t getEFlags(InputFile *file) {
  return cast<ObjFile<ELF64LE>>(file)->getObj().getHeader().e_flags;
}

uint32_t AMDGPU::calcEFlagsV3() const {
  uint32_t ret = getEFlags(ctx.objectFiles[0]);

  // Verify that all input files have the same e_flags.
  for (InputFile *f : ArrayRef(ctx.objectFiles).slice(1)) {
    if (ret == getEFlags(f))
      continue;
    ErrAlways(ctx) << "incompatible e_flags: " << f;
    return 0;
  }
  return ret;
}

uint32_t AMDGPU::calcEFlagsV4() const {
  uint32_t retMach = getEFlags(ctx.objectFiles[0]) & EF_AMDGPU_MACH;
  uint32_t retXnack =
      getEFlags(ctx.objectFiles[0]) & EF_AMDGPU_FEATURE_XNACK_V4;
  uint32_t retSramEcc =
      getEFlags(ctx.objectFiles[0]) & EF_AMDGPU_FEATURE_SRAMECC_V4;

  // Verify that all input files have compatible e_flags (same mach, all
  // features in the same category are either ANY, ANY and ON, or ANY and OFF).
  for (InputFile *f : ArrayRef(ctx.objectFiles).slice(1)) {
    if (retMach != (getEFlags(f) & EF_AMDGPU_MACH)) {
      Err(ctx) << "incompatible mach: " << f;
      return 0;
    }

    if (retXnack == EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4 ||
        (retXnack != EF_AMDGPU_FEATURE_XNACK_ANY_V4 &&
            (getEFlags(f) & EF_AMDGPU_FEATURE_XNACK_V4)
                != EF_AMDGPU_FEATURE_XNACK_ANY_V4)) {
      if (retXnack != (getEFlags(f) & EF_AMDGPU_FEATURE_XNACK_V4)) {
        Err(ctx) << "incompatible xnack: " << f;
        return 0;
      }
    } else {
      if (retXnack == EF_AMDGPU_FEATURE_XNACK_ANY_V4)
        retXnack = getEFlags(f) & EF_AMDGPU_FEATURE_XNACK_V4;
    }

    if (retSramEcc == EF_AMDGPU_FEATURE_SRAMECC_UNSUPPORTED_V4 ||
        (retSramEcc != EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 &&
            (getEFlags(f) & EF_AMDGPU_FEATURE_SRAMECC_V4) !=
                EF_AMDGPU_FEATURE_SRAMECC_ANY_V4)) {
      if (retSramEcc != (getEFlags(f) & EF_AMDGPU_FEATURE_SRAMECC_V4)) {
        Err(ctx) << "incompatible sramecc: " << f;
        return 0;
      }
    } else {
      if (retSramEcc == EF_AMDGPU_FEATURE_SRAMECC_ANY_V4)
        retSramEcc = getEFlags(f) & EF_AMDGPU_FEATURE_SRAMECC_V4;
    }
  }

  return retMach | retXnack | retSramEcc;
}

uint32_t AMDGPU::calcEFlagsV6() const {
  uint32_t flags = calcEFlagsV4();

  uint32_t genericVersion =
      getEFlags(ctx.objectFiles[0]) & EF_AMDGPU_GENERIC_VERSION;

  // Verify that all input files have compatible generic version.
  for (InputFile *f : ArrayRef(ctx.objectFiles).slice(1)) {
    if (genericVersion != (getEFlags(f) & EF_AMDGPU_GENERIC_VERSION)) {
      ErrAlways(ctx) << "incompatible generic version: " << f;
      return 0;
    }
  }

  flags |= genericVersion;
  return flags;
}

uint32_t AMDGPU::calcEFlags() const {
  if (ctx.objectFiles.empty())
    return 0;

  uint8_t abiVersion = cast<ObjFile<ELF64LE>>(ctx.objectFiles[0])
                           ->getObj()
                           .getHeader()
                           .e_ident[EI_ABIVERSION];
  switch (abiVersion) {
  case ELFABIVERSION_AMDGPU_HSA_V2:
  case ELFABIVERSION_AMDGPU_HSA_V3:
    return calcEFlagsV3();
  case ELFABIVERSION_AMDGPU_HSA_V4:
  case ELFABIVERSION_AMDGPU_HSA_V5:
    return calcEFlagsV4();
  case ELFABIVERSION_AMDGPU_HSA_V6:
    return calcEFlagsV6();
  default:
    Err(ctx) << "unknown abi version: " << abiVersion;
    return 0;
  }
}

void AMDGPU::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {
  switch (rel.type) {
  case R_AMDGPU_ABS32:
  case R_AMDGPU_GOTPCREL:
  case R_AMDGPU_GOTPCREL32_LO:
  case R_AMDGPU_REL32:
  case R_AMDGPU_REL32_LO:
  case R_AMDGPU_LDS_OFFSET:
    write32le(loc, val);
    break;
  case R_AMDGPU_ABS64:
  case R_AMDGPU_REL64:
    write64le(loc, val);
    break;
  case R_AMDGPU_GOTPCREL32_HI:
  case R_AMDGPU_REL32_HI:
    write32le(loc, val >> 32);
    break;
  case R_AMDGPU_REL16: {
    int64_t simm = (static_cast<int64_t>(val) - 4) / 4;
    checkInt(ctx, loc, simm, 16, rel);
    write16le(loc, simm);
    break;
  }
  default:
    llvm_unreachable("unknown relocation");
  }
}

RelExpr AMDGPU::getRelExpr(RelType type, const Symbol &s,
                           const uint8_t *loc) const {
  switch (type) {
  case R_AMDGPU_ABS32:
  case R_AMDGPU_ABS64:
    return R_ABS;
  case R_AMDGPU_LDS_OFFSET:
    return RE_AMDGPU_LDS;
  case R_AMDGPU_REL32:
  case R_AMDGPU_REL32_LO:
  case R_AMDGPU_REL32_HI:
  case R_AMDGPU_REL64:
  case R_AMDGPU_REL16:
    return R_PC;
  case R_AMDGPU_GOTPCREL:
  case R_AMDGPU_GOTPCREL32_LO:
  case R_AMDGPU_GOTPCREL32_HI:
    return R_GOT_PC;
  default:
    Err(ctx) << getErrorLoc(ctx, loc) << "unknown relocation (" << type.v
             << ") against symbol " << &s;
    return R_NONE;
  }
}

RelType AMDGPU::getDynRel(RelType type) const {
  if (type == R_AMDGPU_ABS64)
    return type;
  return R_AMDGPU_NONE;
}

int64_t AMDGPU::getImplicitAddend(const uint8_t *buf, RelType type) const {
  switch (type) {
  case R_AMDGPU_NONE:
    return 0;
  case R_AMDGPU_ABS64:
  case R_AMDGPU_RELATIVE64:
    return read64(ctx, buf);
  default:
    InternalErr(ctx, buf) << "cannot read addend for relocation " << type;
    return 0;
  }
}

// Per-function resource usage, parsed from .amdgpu.func_rsrc sections.
struct FuncRsrc {
  uint32_t numVGPRs;
  uint32_t numAGPRs;
  uint32_t numSGPRs;
  uint32_t ldsSize;
  uint32_t scratchSize;
  uint32_t flags;
};

// Patched metadata values for a kernel, used to update .note MsgPack.
struct PatchedKernelMeta {
  uint32_t privateSeg;
  uint32_t groupSeg;
  uint32_t vgprCount;
  uint32_t sgprCount;
  uint32_t agprCount;
  bool usesDynStack;
};

// Find the next occurrence of a MsgPack fixstr key in raw bytes.
// Returns the offset of the fixstr header byte, or SIZE_MAX if not found.
static size_t findFixstrKey(const uint8_t *data, size_t size,
                            llvm::StringRef key, size_t startPos) {
  uint8_t hdr = 0xa0 | static_cast<uint8_t>(key.size());
  for (size_t i = startPos; i + 1 + key.size() <= size; ++i) {
    if (data[i] == hdr &&
        memcmp(data + i + 1, key.data(), key.size()) == 0)
      return i;
  }
  return SIZE_MAX;
}

// Read a MsgPack string value at the given offset and return it.
static llvm::StringRef readMsgPackString(const uint8_t *data, size_t size,
                                         size_t offset) {
  if (offset >= size)
    return {};
  uint8_t b = data[offset];
  if ((b & 0xe0) == 0xa0) {
    size_t len = b & 0x1f;
    if (offset + 1 + len <= size)
      return llvm::StringRef(reinterpret_cast<const char *>(data + offset + 1),
                             len);
  } else if (b == 0xd9 && offset + 1 < size) {
    size_t len = data[offset + 1];
    if (offset + 2 + len <= size)
      return llvm::StringRef(reinterpret_cast<const char *>(data + offset + 2),
                             len);
  }
  return {};
}

// Patch MsgPack metadata fields in place (values pre-widened to uint32).
static void patchNoteMetadata(
    uint8_t *data, size_t size,
    const llvm::DenseMap<llvm::StringRef, PatchedKernelMeta> &patchMap) {

  // Pass 1: collect kernel names in blob order (the ".name" key is sorted
  // before the fields we patch, so we see each kernel's name first).
  SmallVector<llvm::StringRef, 16> kernelNames;
  {
    size_t pos = 0;
    for (;;) {
      pos = findFixstrKey(data, size, ".name", pos);
      if (pos == SIZE_MAX)
        break;
      llvm::StringRef name =
          readMsgPackString(data, size, pos + 1 + 5 /*.name*/);
      if (!name.empty())
        kernelNames.push_back(name);
      pos += 1;
    }
  }

  if (kernelNames.empty())
    return;

  // Helper: patch every occurrence of a uint32-encoded field.
  auto patchUInt32Field = [&](llvm::StringRef key,
                              uint32_t PatchedKernelMeta::*field) {
    size_t pos = 0;
    size_t idx = 0;
    for (;;) {
      pos = findFixstrKey(data, size, key, pos);
      if (pos == SIZE_MAX)
        break;
      size_t valOff = pos + 1 + key.size();
      if (valOff + 5 <= size && data[valOff] == 0xce &&
          idx < kernelNames.size()) {
        auto it = patchMap.find(kernelNames[idx]);
        if (it != patchMap.end()) {
          uint32_t v = it->second.*field;
          data[valOff + 1] = (v >> 24) & 0xff;
          data[valOff + 2] = (v >> 16) & 0xff;
          data[valOff + 3] = (v >> 8) & 0xff;
          data[valOff + 4] = v & 0xff;
        }
      }
      pos += 1;
      ++idx;
    }
  };

  patchUInt32Field(".private_segment_fixed_size",
                   &PatchedKernelMeta::privateSeg);
  patchUInt32Field(".group_segment_fixed_size", &PatchedKernelMeta::groupSeg);
  patchUInt32Field(".vgpr_count", &PatchedKernelMeta::vgprCount);
  patchUInt32Field(".sgpr_count", &PatchedKernelMeta::sgprCount);
  patchUInt32Field(".agpr_count", &PatchedKernelMeta::agprCount);

  // Patch .uses_dynamic_stack (boolean: 0xc3 = true, 0xc2 = false).
  {
    size_t pos = 0;
    size_t idx = 0;
    for (;;) {
      pos = findFixstrKey(data, size, ".uses_dynamic_stack", pos);
      if (pos == SIZE_MAX)
        break;
      size_t valOff = pos + 1 + 19; // strlen(".uses_dynamic_stack")
      if (valOff < size && idx < kernelNames.size()) {
        uint8_t b = data[valOff];
        if (b == 0xc2 || b == 0xc3) {
          auto it = patchMap.find(kernelNames[idx]);
          if (it != patchMap.end())
            data[valOff] = it->second.usesDynStack ? 0xc3 : 0xc2;
        }
      }
      pos += 1;
      ++idx;
    }
  }
}

// GFX90A+ has a unified VGPR/AGPR register file.
static bool hasUnifiedVGPRFile(uint32_t mach) {
  switch (mach) {
  case EF_AMDGPU_MACH_AMDGCN_GFX90A:
  case EF_AMDGPU_MACH_AMDGCN_GFX942:
  case EF_AMDGPU_MACH_AMDGCN_GFX950:
    return true;
  default:
    return false;
  }
}

static uint32_t getTotalNumVGPRs(uint32_t mach, uint32_t numVGPR,
                                 uint32_t numAGPR) {
  if (hasUnifiedVGPRFile(mach) && numAGPR)
    return llvm::alignTo(numVGPR, 4u) + numAGPR;
  return std::max(numVGPR, numAGPR);
}

// Derive the VGPR encoding granularity from the ELF e_flags mach ID and the
// kernel's wave size.  Mirrors AMDGPUBaseInfo::getVGPREncodingGranule() but
// works without MCSubtargetInfo.
static unsigned getVGPREncodingGranule(uint32_t mach, bool isWave32) {
  switch (mach) {
  case EF_AMDGPU_MACH_AMDGCN_GFX90A:
  case EF_AMDGPU_MACH_AMDGCN_GFX942:
  case EF_AMDGPU_MACH_AMDGCN_GFX950:
    return 8;

  case EF_AMDGPU_MACH_AMDGCN_GFX1250:
  case EF_AMDGPU_MACH_AMDGCN_GFX1251:
  case EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC:
    return isWave32 ? 16 : 8;

  default:
    return isWave32 ? 8 : 4;
  }
}

// -fgpu-rdc-isa: patch each kernel descriptor with link-time aggregated
// resource usage (cross-TU callees), replacing per-TU values from each object.
void AMDGPU::postRelocatePass() const {
  using namespace llvm::amdhsa;

  // Step 1: Parse .amdgpu.func_rsrc sections from all input objects.
  // Map from function VA → resource usage.
  llvm::DenseMap<uint64_t, FuncRsrc> funcRsrcMap;

  for (InputFile *file : ctx.objectFiles) {
    for (InputSectionBase *sec : file->getSections()) {
      if (!sec || sec->name != ".amdgpu.func_rsrc")
        continue;

      ArrayRef<uint8_t> data = sec->content();
      if (data.size() < 8)
        continue;

      uint32_t version = read32le(data.data());
      uint32_t numEntries = read32le(data.data() + 4);
      if (version != 1) {
          continue;
      }

      // Each entry: 8 (func_addr) + 6*4 (fields) + 4 (reserved) = 36 bytes
      constexpr size_t entrySize = 36;
      if (data.size() < 8 + numEntries * entrySize) {
        continue;
      }

      // Build offset → VA map from raw relocations (.amdgpu.func_rsrc
      // is non-SHF_ALLOC, so sec->relocs() is empty).
      llvm::DenseMap<uint64_t, uint64_t> offsetToVA;
      const RelsOrRelas<ELF64LE> rawRels =
          sec->template relsOrRelas<ELF64LE>();
      for (const typename ELF64LE::Rela &rel : rawRels.relas) {
        uint32_t symIdx = rel.getSymbol(false);
        Symbol &sym = file->getSymbol(symIdx);
        if (auto *def = dyn_cast<Defined>(&sym))
          offsetToVA[rel.r_offset] = def->getVA(ctx);
      }
      for (const typename ELF64LE::Rel &rel : rawRels.rels) {
        uint32_t symIdx = rel.getSymbol(false);
        Symbol &sym = file->getSymbol(symIdx);
        if (auto *def = dyn_cast<Defined>(&sym))
          offsetToVA[rel.r_offset] = def->getVA(ctx);
      }

      for (uint32_t i = 0; i < numEntries; ++i) {
        size_t off = 8 + i * entrySize;
        const uint8_t *entry = data.data() + off;

        auto vaIt = offsetToVA.find(off);
        if (vaIt == offsetToVA.end() || !vaIt->second)
          continue;
        uint64_t funcVA = vaIt->second;

        FuncRsrc rsrc;
        rsrc.numVGPRs = read32le(entry + 8);
        rsrc.numAGPRs = read32le(entry + 12);
        rsrc.numSGPRs = read32le(entry + 16);
        rsrc.ldsSize = read32le(entry + 20);
        rsrc.scratchSize = read32le(entry + 24);
        rsrc.flags = read32le(entry + 28);
        funcRsrcMap[funcVA] = rsrc;
      }
    }
  }

  // Step 2: Build call graph from relocations in executable sections.
  // Map from caller VA → set of callee VAs.
  llvm::DenseMap<uint64_t, SmallVector<uint64_t, 4>> callGraph;

  for (InputFile *file : ctx.objectFiles) {
    for (InputSectionBase *sec : file->getSections()) {
      if (!sec || !(sec->flags & llvm::ELF::SHF_EXECINSTR))
        continue;

      for (const Relocation &rel : sec->relocs()) {
        // PC-relative relocations in code are typically function calls.
        if (rel.type != R_AMDGPU_REL32 && rel.type != R_AMDGPU_REL32_LO &&
            rel.type != R_AMDGPU_REL64)
          continue;
        if (!rel.sym || rel.sym->isUndefined())
          continue;

        uint64_t callSiteVA =
            sec->getVA(rel.offset);
        uint64_t calleeVA = rel.sym->getVA(ctx, rel.addend);

        // Attribute the call site to a function by finding the function
        // whose range contains the call site. Walk the function resource
        // map — entries whose VA <= callSiteVA.
        uint64_t callerVA = 0;
        for (auto &[va, rsrc] : funcRsrcMap) {
          if (va <= callSiteVA && va > callerVA)
            callerVA = va;
        }

        if (callerVA && calleeVA && callerVA != calleeVA)
          callGraph[callerVA].push_back(calleeVA);
      }
    }
  }

  // Step 3: Collect kernel descriptors.
  struct KDInfo {
    Defined *sym;
    uint8_t *buf;
    uint64_t kernelVA; // VA of the kernel function (not the .kd)
  };
  SmallVector<KDInfo, 16> kernelDescriptors;

  for (InputFile *file : ctx.objectFiles) {
    for (Symbol *sym : file->getSymbols()) {
      auto *def = dyn_cast_or_null<Defined>(sym);
      if (!def || !def->section)
        continue;
      if (!def->getName().ends_with(".kd"))
        continue;
      if (def->size != sizeof(kernel_descriptor_t))
        continue;

      auto *isec = dyn_cast<InputSection>(def->section);
      if (!isec || !isec->getParent())
        continue;

      OutputSection *osec = isec->getParent();
      uint64_t fileOff = osec->offset + (def->getVA(ctx) - osec->addr);
      uint8_t *buf = ctx.bufferStart + fileOff;

      // The kernel function symbol is the .kd name without the .kd suffix.
      StringRef kdName = def->getName();
      StringRef kernelName = kdName.drop_back(3); // remove ".kd"
      uint64_t kernelVA = 0;
      if (Symbol *kernSym = ctx.symtab->find(kernelName)) {
        if (auto *kernDef = dyn_cast<Defined>(kernSym))
          kernelVA = kernDef->getVA(ctx);
      }

      kernelDescriptors.push_back({def, buf, kernelVA});
    }
  }

  if (kernelDescriptors.empty())
    return;

  // Read the target mach from e_flags for VGPR encoding granularity.
  uint32_t mach = 0;
  if (!ctx.objectFiles.empty())
    mach = getEFlags(ctx.objectFiles[0]) & EF_AMDGPU_MACH;

  // Global max registers/LDS over device functions only. Kernel entries
  // already account for their own call tree, so including them in the
  // indirect-call fallback used below would double-count and reduce
  // occupancy.
  FuncRsrc globalMax = {0, 0, 0, 0, 0, 0};
  bool globalHasDynStack = false;
  for (auto &[va, rsrc] : funcRsrcMap) {
    if (rsrc.flags & (1 << 4)) // IsKernel
      continue;
    globalMax.numVGPRs = std::max(globalMax.numVGPRs, rsrc.numVGPRs);
    globalMax.numAGPRs = std::max(globalMax.numAGPRs, rsrc.numAGPRs);
    globalMax.numSGPRs = std::max(globalMax.numSGPRs, rsrc.numSGPRs);
    globalMax.ldsSize = std::max(globalMax.ldsSize, rsrc.ldsSize);
    globalMax.scratchSize = std::max(globalMax.scratchSize, rsrc.scratchSize);
    if (rsrc.flags & (1 << 2))
      globalHasDynStack = true;
  }

  // Compute total scratch per function via memoized DFS over the call graph.
  llvm::DenseMap<uint64_t, uint32_t> totalScratchMemo;
  llvm::DenseSet<uint64_t> scratchInProgress; // cycle detection

  std::function<uint32_t(uint64_t)> computeTotalScratch =
      [&](uint64_t va) -> uint32_t {
    auto memoIt = totalScratchMemo.find(va);
    if (memoIt != totalScratchMemo.end())
      return memoIt->second;

    if (!scratchInProgress.insert(va).second)
      return 0; // cycle — break it

    uint32_t ownScratch = 0;
    auto it = funcRsrcMap.find(va);
    if (it != funcRsrcMap.end())
      ownScratch = it->second.scratchSize;

    uint32_t maxCalleeScratch = 0;
    auto cgIt = callGraph.find(va);
    if (cgIt != callGraph.end()) {
      for (uint64_t callee : cgIt->second)
        maxCalleeScratch =
            std::max(maxCalleeScratch, computeTotalScratch(callee));
    }

    uint32_t total = ownScratch + maxCalleeScratch;
    totalScratchMemo[va] = total;
    scratchInProgress.erase(va);
    return total;
  };

  // Precompute total scratch for every function.
  uint32_t globalMaxTotalScratch = 0;
  for (auto &[va, rsrc] : funcRsrcMap) {
    uint32_t ts = computeTotalScratch(va);
    globalMaxTotalScratch = std::max(globalMaxTotalScratch, ts);
  }

  // Diagnostic summary of the aggregated resource usage that will be patched
  // into the kernel descriptors. Skipped when there is nothing to patch.
  if (!funcRsrcMap.empty()) {
    size_t devfuncCount = 0;
    for (auto &[va, rsrc] : funcRsrcMap)
      if (!(rsrc.flags & (1 << 4)))
        ++devfuncCount;

    uint32_t archVGPR = globalMax.numVGPRs;
    uint32_t agpr = globalMax.numAGPRs;
    uint32_t accumOffset = llvm::alignTo(archVGPR, 4u);
    uint32_t totalVGPR = getTotalNumVGPRs(mach, archVGPR, agpr);
    uint32_t nextFreeVGPR =
        hasUnifiedVGPRFile(mach) ? std::max(totalVGPR, accumOffset + agpr)
                                 : totalVGPR;

    Msg(ctx) << "[amdgpu-rdc-isa] Aggregated " << devfuncCount
             << " device functions (of " << funcRsrcMap.size()
             << " entries): vgpr_count=" << totalVGPR
             << ", agpr_count=" << agpr
             << ", sgpr_count=" << globalMax.numSGPRs
             << ", scratch=" << globalMaxTotalScratch
             << ", group_segment=" << globalMax.ldsSize
             << ", accum_offset=" << accumOffset
             << ", next_free_vgpr=" << nextFreeVGPR
             << " [arch_vgpr=" << archVGPR
             << ", kernels=" << kernelDescriptors.size() << "]";
  }

  // Step 4: For each kernel, walk the call graph to compute cross-TU
  // resource aggregates, then patch the kernel descriptor.
  llvm::DenseMap<llvm::StringRef, PatchedKernelMeta> patchMap;

  for (KDInfo &kd : kernelDescriptors) {
    // DFS to find all reachable functions — collect max regs/LDS and
    // total scratch (accumulated along call depth).
    SmallVector<uint64_t, 16> worklist;
    llvm::DenseSet<uint64_t> visited;

    if (kd.kernelVA) {
      worklist.push_back(kd.kernelVA);
      visited.insert(kd.kernelVA);
    }

    uint32_t maxVGPRs = 0, maxAGPRs = 0, maxSGPRs = 0;
    uint32_t maxLDS = 0;
    bool hasIndirectCall = false;
    bool hasDynSizedStack = false;

    while (!worklist.empty()) {
      uint64_t va = worklist.pop_back_val();
      auto it = funcRsrcMap.find(va);
      if (it != funcRsrcMap.end()) {
        const FuncRsrc &r = it->second;
        maxVGPRs = std::max(maxVGPRs, r.numVGPRs);
        maxAGPRs = std::max(maxAGPRs, r.numAGPRs);
        maxSGPRs = std::max(maxSGPRs, r.numSGPRs);
        maxLDS = std::max(maxLDS, r.ldsSize);
        if (r.flags & (1 << 3))
          hasIndirectCall = true;
        if (r.flags & (1 << 2))
          hasDynSizedStack = true;
      }

      auto cgIt = callGraph.find(va);
      if (cgIt != callGraph.end()) {
        for (uint64_t callee : cgIt->second) {
          if (visited.insert(callee).second)
            worklist.push_back(callee);
        }
      }
    }

    // Scratch: use accumulated total from the kernel's call tree.
    uint32_t maxScratch = kd.kernelVA ? computeTotalScratch(kd.kernelVA) : 0;

    // Fall back to global max if no call edges or indirect calls present.
    bool useGlobalMax = (visited.size() <= 1 && callGraph.count(kd.kernelVA) == 0)
                        || hasIndirectCall;
    if (useGlobalMax) {
      maxVGPRs = std::max(maxVGPRs, globalMax.numVGPRs);
      maxAGPRs = std::max(maxAGPRs, globalMax.numAGPRs);
      maxSGPRs = std::max(maxSGPRs, globalMax.numSGPRs);
      maxLDS = std::max(maxLDS, globalMax.ldsSize);
      maxScratch = std::max(maxScratch, globalMaxTotalScratch);
      hasDynSizedStack = hasDynSizedStack || globalHasDynStack;
    }

    // Several rsrc2/kcProps bits gate user-SGPR / runtime setup that the
    // kernel prologue must match. They may only be cleared post-link, not
    // set 0->1, or HSA will reject the code object.
    uint16_t kcProps = read16le(kd.buf + KERNEL_CODE_PROPERTIES_OFFSET);
    uint32_t rsrc2 = read32le(kd.buf + COMPUTE_PGM_RSRC2_OFFSET);
    bool kdHasPrivateSegment =
        AMDHSA_BITS_GET(rsrc2, COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT);
    bool kdHasDynStack =
        AMDHSA_BITS_GET(kcProps, KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK);

    // Patch the kernel descriptor in the output buffer.
    if (maxLDS > read32le(kd.buf + GROUP_SEGMENT_FIXED_SIZE_OFFSET))
      write32le(kd.buf + GROUP_SEGMENT_FIXED_SIZE_OFFSET, maxLDS);
    // Only grow the scratch byte count when ENABLE_PRIVATE_SEGMENT is
    // already set; otherwise the prologue and user-SGPRs are not set up to
    // access scratch and growing the value would be unsafe.
    if (kdHasPrivateSegment &&
        maxScratch > read32le(kd.buf + PRIVATE_SEGMENT_FIXED_SIZE_OFFSET))
      write32le(kd.buf + PRIVATE_SEGMENT_FIXED_SIZE_OFFSET, maxScratch);

    bool isWave32 = AMDHSA_BITS_GET(kcProps,
        KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);

    unsigned vgprGranularity = getVGPREncodingGranule(mach, isWave32);
    constexpr unsigned sgprGranularity = 8; // constant across all targets

    // Patch VGPR/SGPR counts in compute_pgm_rsrc1.
    uint32_t rsrc1 = read32le(kd.buf + COMPUTE_PGM_RSRC1_OFFSET);
    uint32_t curVGPRBlocks = AMDHSA_BITS_GET(
        rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT);
    uint32_t curSGPRBlocks = AMDHSA_BITS_GET(
        rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);

    uint32_t totalVGPRs = getTotalNumVGPRs(mach, maxVGPRs, maxAGPRs);
    uint32_t newVGPRBlocks =
        (std::max(1u, (totalVGPRs + vgprGranularity - 1) /
                           vgprGranularity)) -
        1;

    if (newVGPRBlocks > curVGPRBlocks) {
      AMDHSA_BITS_SET(rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                      newVGPRBlocks);
    }

    uint32_t newSGPRBlocks =
        (std::max(1u, (maxSGPRs + sgprGranularity - 1) / sgprGranularity)) - 1;
    if (newSGPRBlocks > curSGPRBlocks) {
      AMDHSA_BITS_SET(rsrc1, COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                      newSGPRBlocks);
    }

    write32le(kd.buf + COMPUTE_PGM_RSRC1_OFFSET, rsrc1);

    // Patch ACCUM_OFFSET in RSRC3 (ArchVGPR/AccVGPR boundary) for GFX90A+.
    if (hasUnifiedVGPRFile(mach) && maxAGPRs > 0) {
      uint32_t rsrc3 = read32le(kd.buf + COMPUTE_PGM_RSRC3_OFFSET);
      uint32_t curAccumOff = AMDHSA_BITS_GET(rsrc3,
          COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET);
      uint32_t newAccumOff = (llvm::alignTo(maxVGPRs, 4u) / 4) - 1;
      if (newAccumOff > curAccumOff) {
        AMDHSA_BITS_SET(rsrc3, COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET,
                        newAccumOff);
        write32le(kd.buf + COMPUTE_PGM_RSRC3_OFFSET, rsrc3);
      }
    }

    // Clear USES_DYNAMIC_STACK if the aggregated call tree doesn't need it.
    // We never set this bit 0->1 because the prologue setup it requires
    // cannot be added post-link.
    if (kdHasDynStack && !hasDynSizedStack) {
      AMDHSA_BITS_SET(kcProps, KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK, 0);
      write16le(kd.buf + KERNEL_CODE_PROPERTIES_OFFSET, kcProps);
    }

    // Clear ENABLE_PRIVATE_SEGMENT when no aggregated callee uses scratch,
    // mirroring the same "clear only" constraint.
    if (kdHasPrivateSegment && maxScratch == 0) {
      AMDHSA_BITS_SET(rsrc2, COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 0);
      write32le(kd.buf + COMPUTE_PGM_RSRC2_OFFSET, rsrc2);
    }

    // Record patched values for .note metadata patching (Step 5).
    StringRef kernelName = kd.sym->getName().drop_back(3); // remove ".kd"
    patchMap[kernelName] = {
        read32le(kd.buf + PRIVATE_SEGMENT_FIXED_SIZE_OFFSET),
        read32le(kd.buf + GROUP_SEGMENT_FIXED_SIZE_OFFSET),
        maxVGPRs, maxSGPRs, maxAGPRs, hasDynSizedStack};
  }

  // Step 5: Patch NT_AMDGPU_METADATA notes to match the patched KDs.
  for (InputFile *file : ctx.objectFiles) {
    for (InputSectionBase *sec : file->getSections()) {
      if (!sec || sec->type != llvm::ELF::SHT_NOTE ||
          !(sec->flags & llvm::ELF::SHF_ALLOC))
        continue;
      auto *isec = dyn_cast<InputSection>(sec);
      if (!isec || !isec->getParent())
        continue;

      OutputSection *osec = isec->getParent();
      uint64_t fileOff = osec->offset + isec->outSecOff;
      uint8_t *noteBase = ctx.bufferStart + fileOff;
      size_t noteSize = isec->getSize();

      size_t pos = 0;
      while (pos + 12 <= noteSize) {
        uint32_t namesz = read32le(noteBase + pos);
        uint32_t descsz = read32le(noteBase + pos + 4);
        uint32_t type = read32le(noteBase + pos + 8);

        size_t nameAligned = llvm::alignTo(namesz, 4u);
        size_t descOff = pos + 12 + nameAligned;

        if (descOff + descsz > noteSize)
          break;

        if (type == llvm::ELF::NT_AMDGPU_METADATA)
          patchNoteMetadata(noteBase + descOff, descsz, patchMap);

        pos = descOff + llvm::alignTo(descsz, 4u);
      }
    }
  }
}

void elf::setAMDGPUTargetInfo(Ctx &ctx) { ctx.target.reset(new AMDGPU(ctx)); }
