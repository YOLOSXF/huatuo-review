#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
肝病试题筛选脚本 - 专业版关键词库
从医学测评数据集中筛选肝病相关试题，保持原有格式
"""

import json
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import re

# 数据集键名（根据你的描述）
DATASET_KEYS = [
    "MedMCQA_validation",
    "MedQA_USLME_test",
    "PubMedQA_test",
    "MMLU-Pro_Medical_test",
    "GPQA_Medical_test"
]

# 专业版肝病关键词库（131个专业术语）
LIVER_DISEASE_KEYWORDS = [
    # 基础肝脏疾病
    'liver', 'hepatic', 'hepatitis', 'hepatology', 'hepatocyte',
    'hepatotoxic', 'hepatorenal', 'hepatopulmonary',

    # 特定疾病
    'cirrhosis', 'hepatocellular', 'hepatoma', 'hepatoblastoma',
    'cholangitis', 'cholangiocarcinoma', 'cholestasis', 'cholestatic',
    'bilirubin', 'hyperbilirubinemia', 'jaundice', 'icterus',
    'steatosis', 'steatohepatitis', 'fibrosis', 'sclerosing cholangitis',

    # 并发症
    'hepatic encephalopathy', 'portal hypertension', 'portosystemic',
    'varices', 'variceal', 'ascites', 'hepatomegaly', 'splenomegaly',
    'hepatosplenomegaly', 'fetor hepaticus', 'asterixis',

    # 病毒性肝炎
    'viral hepatitis', 'hbv', 'hcv', 'hdv', 'hev', 'hav',
    'hepatitis b', 'hepatitis c', 'hepatitis a', 'hepatitis d', 'hepatitis e',
    'hb surface antigen', 'hcv rna', 'hbv dna',

    # 酒精/代谢性肝病
    'alcoholic liver', 'alcoholic hepatitis', 'alcoholic cirrhosis',
    'non-alcoholic', 'nafld', 'nash', 'fatty liver',

    # 自身免疫性肝病
    'autoimmune hepatitis', 'primary biliary', 'pbc', 'psc',
    'anti-mitochondrial', 'anti-smooth muscle', 'lkm antibodies',

    # 遗传/代谢性疾病
    'wilson disease', 'hemochromatosis', 'alpha-1 antitrypsin',
    'gilbert syndrome', 'crigler-najjar', 'dubin-johnson', 'rotor syndrome',
    'galactosemia', 'fructose intolerance', 'tyrosinemia',

    # 血管性疾病
    'budd-chiari', 'hepatic vein', 'portal vein', 'veno-occlusive',
    'sinusoidal obstruction',

    # 胆道相关（你如果只要“肝脏疾病”，可以把这一块整体降权/移除）
    'biliary', 'bile duct', 'gallbladder', 'gall stone', 'cholelithiasis',
    'cholecystitis', 'biliary atresia', 'caroli disease',

    # 药物/中毒
    'drug-induced liver', 'acetaminophen toxicity', 'paracetamol overdose',
    'toxic hepatitis', 'fulminant hepatic failure',

    # 治疗相关
    'liver transplant', 'liver biopsy', 'tips', 'paracentesis',
    'transjugular', 'hepatic resection',

    # 肿瘤（良性+恶性）
    'liver cancer', 'hcc', 'hepatoblastoma', 'hemangioma', 'fnh',
    'hepatic adenoma', 'hepatocellular carcinoma',

    # 中文关键词
    '肝', '肝炎', '肝硬化', '肝癌', '肝细胞', '肝功能',
    '肝病', '肝脏', '胆汁', '黄疸', '腹水', '门脉高压',
    '脂肪肝', '酒精肝', '自身免疫性肝炎', '病毒性肝炎',
    '肝性脑病', '肝移植', '胆道', '胆囊', '胆结石', '胆囊炎',
    '甲肝', '乙肝', '丙肝', '丁肝', '戊肝'
]

# -----------------------------
# 核心词 + 阈值：避免数据污染
# -----------------------------
CORE_KEYWORDS = {
    # 强相关：疾病/关键诊断实体/关键并发症
    "hepatitis", "viral hepatitis", "hepatitis a", "hepatitis b", "hepatitis c", "hepatitis d", "hepatitis e",
    "hbv", "hcv", "hdv", "hev", "hav", "hb surface antigen", "hcv rna", "hbv dna",
    "cirrhosis", "portal hypertension", "ascites", "hepatic encephalopathy",
    "bilirubin", "hyperbilirubinemia", "jaundice", "icterus",
    "nafld", "nash", "steatohepatitis", "fatty liver", "steatosis",
    "autoimmune hepatitis", "pbc", "psc", "primary biliary", "sclerosing cholangitis",
    "wilson disease", "hemochromatosis", "alpha-1 antitrypsin",
    "budd-chiari", "fulminant hepatic failure", "drug-induced liver", "toxic hepatitis",
    "hepatocellular carcinoma", "hepatocellular", "hepatoma", "liver cancer", "hcc",

    # 中文强相关
    "肝炎", "病毒性肝炎", "甲肝", "乙肝", "丙肝", "丁肝", "戊肝",
    "肝硬化", "门脉高压", "腹水", "肝性脑病",
    "黄疸", "胆红素",  # “胆红素”不在原关键词里，但可作为强相关（不影响匹配逻辑，仅用于评分判定时的命中需要在关键词表里）
    "脂肪肝", "酒精肝",
    "自身免疫性肝炎",
    "肝癌", "肝细胞癌",
    "肝衰竭", "暴发性肝衰竭", "药物性肝损伤"
}

# 弱相关：解剖/泛化词/检查或治疗/胆道泛化（容易误召回）
WEAK_KEYWORDS = {
    "liver", "hepatic", "hepatology", "hepatocyte",
    "hepatomegaly", "splenomegaly", "hepatosplenomegaly",
    "liver biopsy", "liver transplant", "hepatic resection", "tips", "paracentesis", "transjugular",
    "biliary", "bile duct", "gallbladder", "gall stone", "cholelithiasis", "cholecystitis", "biliary atresia",
    "肝", "肝脏", "肝功能", "肝病", "肝移植", "胆汁", "胆道", "胆囊", "胆结石", "胆囊炎"
}

_WORD_RE_CACHE: dict[str, re.Pattern] = {}

def _is_ascii_word(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]+", s))

def _contains_keyword(text_lower: str, keyword: str) -> bool:
    """
    关键词匹配：
    - 英文“单词”用词边界，避免 deliver/silver 命中 liver
    - 其他（中文/短语/带空格）用子串
    """
    kw_lower = keyword.lower()
    if _is_ascii_word(keyword):
        pat = _WORD_RE_CACHE.get(kw_lower)
        if pat is None:
            pat = re.compile(rf"\b{re.escape(kw_lower)}\b")
            _WORD_RE_CACHE[kw_lower] = pat
        return pat.search(text_lower) is not None
    return kw_lower in text_lower

def is_liver_related(text: str, *, min_score: int = 2, require_core: bool = True) -> Tuple[bool, List[str]]:
    """
    核心词 + 阈值的肝病相关判断

    计分规则：
      - 命中 CORE_KEYWORDS：+2
      - 命中 WEAK_KEYWORDS：+1
      - 其他关键词（在 LIVER_DISEASE_KEYWORDS 里但不在 CORE/WEAK）：+1

    通过条件（默认）：
      score >= 2 且至少命中 1 个核心词（require_core=True）
    """
    if not isinstance(text, str) or not text.strip():
        return False, []

    text_lower = text.lower()
    matched_keywords: list[str] = []
    score = 0
    core_hits = 0

    for keyword in LIVER_DISEASE_KEYWORDS:
        if not keyword or not isinstance(keyword, str):
            continue
        if _contains_keyword(text_lower, keyword):
            matched_keywords.append(keyword)

            kw_lower = keyword.lower()
            if kw_lower in {k.lower() for k in CORE_KEYWORDS}:
                score += 2
                core_hits += 1
            elif kw_lower in {k.lower() for k in WEAK_KEYWORDS}:
                score += 1
            else:
                score += 1

    matched_keywords = sorted(set(matched_keywords))
    passed = (score >= min_score) and ((core_hits > 0) or (not require_core))
    return passed, matched_keywords


def extract_all_text(item: Dict[str, Any]) -> str:
    """
    提取试题中的所有文本内容（问题+选项+答案）
    """
    texts = []

    question = item.get('question', '')
    if isinstance(question, str):
        texts.append(question)

    options = item.get('options', {})
    if isinstance(options, dict):
        for opt_text in options.values():
            if isinstance(opt_text, str):
                texts.append(opt_text)

    answer = item.get('answer', '')
    if isinstance(answer, str):
        texts.append(answer)

    answer_idx = item.get('answer_idx', '')
    if answer_idx and isinstance(options, dict) and answer_idx in options:
        texts.append(options[answer_idx])

    return ' '.join(texts)


def filter_single_question(item: Dict[str, Any], *, min_score: int, require_core: bool) -> Tuple[bool, List[str]]:
    """
    判断单个试题是否与肝病相关
    """
    full_text = extract_all_text(item)
    is_liver, keywords = is_liver_related(full_text, min_score=min_score, require_core=require_core)
    return is_liver, keywords


def categorize_keywords(keywords: List[str]) -> Dict[str, List[str]]:
    """
    将匹配到的关键词按类别分类
    """
    categories = {
        '基础/解剖': ['liver', 'hepatic', 'hepatocyte', '肝', '肝脏'],
        '炎症/感染': ['hepatitis', '肝炎', 'cholangitis', 'cholecystitis'],
        '肝硬化/纤维化': ['cirrhosis', 'fibrosis', '肝硬化'],
        '肿瘤': ['hepatoma', 'hcc', 'carcinoma', 'cancer', '肝癌', 'hemangioma', 'adenoma'],
        '代谢/酒精': ['steatosis', 'nafld', 'nash', 'alcoholic', '脂肪肝', '酒精肝'],
        '自身免疫': ['autoimmune', 'pbc', 'psc'],
        '并发症': ['encephalopathy', 'hypertension', 'ascites', 'varices'],
        '其他': []
    }

    result = defaultdict(list)
    for kw in keywords:
        kw_lower = kw.lower()
        found = False
        for cat, cat_keywords in categories.items():
            if any(cat_kw.lower() in kw_lower for cat_kw in cat_keywords):
                result[cat].append(kw)
                found = True
                break
        if not found:
            result['其他'].append(kw)

    return dict(result)


def filter_dataset_questions(
    questions: List[Dict],
    dataset_name: str,
    verbose: bool = False,
    *,
    min_score: int = 2,
    require_core: bool = True,
    add_match_metadata: bool = False
) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    将单个数据集拆分为：
      - liver_questions：肝病相关
      - non_liver_questions：非肝病（其他疾病/其他题目全部保留）
    """
    liver_questions: List[Dict] = []
    non_liver_questions: List[Dict] = []
    all_matched_keywords: List[str] = []

    stats = {
        'dataset': dataset_name,
        'total': len(questions),
        'matched_liver': 0,
        'matched_non_liver': 0,
        'matched_keywords': set(),
        'categories': defaultdict(int),
        'min_score': min_score,
        'require_core': require_core,
    }

    for item in questions:
        is_liver, keywords = filter_single_question(item, min_score=min_score, require_core=require_core)

        if is_liver:
            item_copy = item.copy()
            if add_match_metadata:
                item_copy['_matched_keywords'] = keywords
                item_copy['_keyword_categories'] = categorize_keywords(keywords)
            liver_questions.append(item_copy)

            stats['matched_liver'] += 1
            stats['matched_keywords'].update(keywords)
            all_matched_keywords.extend(keywords)
        else:
            # 非肝病题：原样保留（不添加任何字段，除非你也想 debug）
            item_copy = item.copy()
            if add_match_metadata:
                item_copy['_matched_keywords'] = []
                item_copy['_keyword_categories'] = {}
            non_liver_questions.append(item_copy)
            stats['matched_non_liver'] += 1

    for kw in all_matched_keywords:
        cats = categorize_keywords([kw])
        for cat in cats:
            stats['categories'][cat] += 1

    stats['matched_keywords'] = sorted(list(stats['matched_keywords']))
    stats['match_ratio_liver'] = stats['matched_liver'] / stats['total'] * 100 if stats['total'] > 0 else 0
    stats['categories'] = dict(stats['categories'])

    return liver_questions, non_liver_questions, stats


def process_medical_dataset(
    input_path: str,
    output_path: str,
    verbose: bool = False,
    *,
    min_score: int = 2,
    require_core: bool = True,
    add_match_metadata: bool = False,
    output_non_liver_path: str | None = None,
) -> Tuple[Dict, Dict, List[Dict]]:
    """
    处理医学测评数据集，输出两份文件：
      1) 肝病测评集（output_path）
      2) 非肝病测评集（output_non_liver_path）
    两份都保持原始结构不变：dict[str, list[dict]]
    """
    print(f"正在加载数据: {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("数据格式错误：期望是一个包含多个数据集的字典")

    if output_non_liver_path is None:
        if output_path.endswith(".json"):
            output_non_liver_path = output_path[:-5] + "_non_liver.json"
        else:
            output_non_liver_path = output_path + "_non_liver.json"

    filtered_liver: Dict[str, Any] = {}
    filtered_non_liver: Dict[str, Any] = {}
    all_stats: List[Dict] = []

    print(f"发现 {len([k for k in data.keys() if not k.startswith('_')])} 个测评子集")
    print("=" * 80)

    for key in data.keys():
        if key.startswith('_'):
            continue

        if key not in DATASET_KEYS:
            print(f"\n⚠️  警告: 发现未预定义数据集 '{key}'，仍将处理")

        subset = data[key]
        if not isinstance(subset, list):
            print(f"\n⚠️  警告: '{key}' 不是列表，将跳过")
            continue

        print(f"\n📊 处理数据集: {key}")
        print(f"   原始试题数: {len(subset)}")

        liver_questions, non_liver_questions, stats = filter_dataset_questions(
            subset, key, verbose=verbose,
            min_score=min_score, require_core=require_core, add_match_metadata=add_match_metadata
        )

        filtered_liver[key] = liver_questions
        filtered_non_liver[key] = non_liver_questions
        all_stats.append(stats)

        print(f"   ✅ 肝病相关: {stats['matched_liver']} 题 ({stats['match_ratio_liver']:.2f}%)")
        print(f"   ✅ 非肝病题: {stats['matched_non_liver']} 题")

        if verbose and stats['matched_liver'] > 0:
            print(f"   🔍 肝病题主要关键词类别:")
            for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1])[:3]:
                print(f"      - {cat}: {count}次")

    # 两份都写 metadata（避免下游误用时不知道规则）
    meta_common = {
        'original_file': input_path,
        'filter_keywords_count': len(LIVER_DISEASE_KEYWORDS),
        'datasets_processed': len(all_stats),
        'description': 'Split into liver vs non-liver question sets',
        'filter_version': '3.1 (split liver/non-liver, core+threshold)',
        'min_score': min_score,
        'require_core': require_core,
    }
    filtered_liver['_filter_metadata'] = {**meta_common, 'split': 'liver'}
    filtered_non_liver['_filter_metadata'] = {**meta_common, 'split': 'non_liver'}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_liver, f, ensure_ascii=False, indent=2)
    with open(output_non_liver_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_non_liver, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 80)
    print(f"✅ 筛选完成！")
    print(f"   肝病测评集: {output_path}")
    print(f"   非肝病测评集: {output_non_liver_path}")

    return filtered_liver, filtered_non_liver, all_stats


def main():
    parser = argparse.ArgumentParser(
        description='从医学测评数据集中筛选肝病相关试题（使用专业版关键词库）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('-i', '--input', required=False,
                        help='输入JSON文件路径',
                        default='/jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/evaluation/data/eval_data.json')
    parser.add_argument('-o', '--output', required=False,
                        default='/jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/evaluation/data/eval_filtered.json',
                        help='输出【肝病测评集】JSON文件路径')
    parser.add_argument('--output-non-liver', required=False, default='/jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/evaluation/data/eval_filtered_other.json',
                        help='输出【非肝病测评集】JSON文件路径（默认在 output 基础上自动加 _non_liver.json）')

    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细匹配信息（包括关键词分类统计）')
    parser.add_argument('--no-metadata', action='store_true',
                        help='不添加_filter_metadata字段到输出')

    # 新增：核心词+阈值控制
    parser.add_argument('--min-score', type=int, default=2,
                        help='通过筛选所需的最小得分（CORE=2分，WEAK=1分；默认2）')
    parser.add_argument('--no-require-core', action='store_true',
                        help='不强制命中核心词（不推荐用于直接构建训练/评测集）')
    parser.add_argument('--add-match-metadata', action='store_true',
                        help='在每条样本中添加 _matched_keywords/_keyword_categories（用于debug；会改变原始结构）')

    args = parser.parse_args()

    print("=" * 80)
    print("医学测评数据集 - 肝病试题筛选工具（核心词+阈值 + 拆分版）")
    print("=" * 80)
    print(f"关键词库: {len(LIVER_DISEASE_KEYWORDS)} 个")
    print(f"规则: min_score={args.min_score}, require_core={not args.no_require_core}")

    try:
        filtered_liver, filtered_non_liver, stats = process_medical_dataset(
            args.input, args.output, args.verbose,
            min_score=args.min_score,
            require_core=(not args.no_require_core),
            add_match_metadata=args.add_match_metadata,
            output_non_liver_path=args.output_non_liver,
        )

        if args.no_metadata:
            if '_filter_metadata' in filtered_liver:
                del filtered_liver['_filter_metadata']
            if '_filter_metadata' in filtered_non_liver:
                del filtered_non_liver['_filter_metadata']

            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(filtered_liver, f, ensure_ascii=False, indent=2)

            non_path = args.output_non_liver
            if non_path is None:
                non_path = (args.output[:-5] + "_non_liver.json") if args.output.endswith(".json") else (args.output + "_non_liver.json")
            with open(non_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_non_liver, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()