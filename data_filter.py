import json
import re

# ====== 精准肝病关键词库（三重保障设计） ======
LIVER_KEYWORDS_CN = {
    # 核心病名（高置信度，优先匹配）
    "core_diseases": [
        "病毒性肝炎", "乙肝", "丙肝", "甲肝", "戊肝", "丁肝", "脂肪肝", "酒精性肝病", 
        "药物性肝损伤", "自身免疫性肝炎", "原发性胆汁性胆管炎", "肝硬化", "肝癌", 
        "肝纤维化", "肝衰竭", "肝脓肿", "肝豆状核变性", "布加综合征", "肝囊肿"
    ],
    # 肝脏特异性检查/指标（排除非肝源性症状干扰）
    "liver_specific": [
        "肝功能", "alt", "ast", "ggt", "alp", "胆红素", "白蛋白", "凝血酶原时间", 
        "肝脏超声", "肝脏b超", "肝脏ct", "肝脏mri", "肝穿刺", "肝活检", "fibroscan",
        "child-pugh", "meld评分", "肝硬度"
    ],
    # 中医肝病术语（避免泛化"肝"字）
    "tcm": [
        "肝郁", "肝气郁结", "肝阳上亢", "肝火上炎", "肝血虚", "肝阴虚", 
        "肝风内动", "肝胆湿热", "疏肝", "清肝利胆", "养肝柔肝"
    ]
}

LIVER_KEYWORDS_EN = {
    # =============== 高置信度（单独出现即保留）===============
    "high_confidence": [
        # 疾病命名（含最新术语）
        "hepatitis B", "HBV", "hepatitis C", "HCV", "hepatitis virus", 
        "cirrhosis", "decompensated cirrhosis", "compensated cirrhosis",
        "hepatocellular carcinoma", "HCC", "liver cancer", "cholangiocarcinoma",
        "MASLD", "MASH", "NAFLD", "NASH", "metabolic dysfunction-associated steatotic liver disease",
        "alcoholic liver disease", "alcoholic hepatitis", "ASH",
        "autoimmune hepatitis", "primary biliary cholangitis", "PBC",
        "primary sclerosing cholangitis", "PSC", "Wilson disease", "hemochromatosis",
        "alpha-1 antitrypsin deficiency", "alagille syndrome", "biliary atresia",
        "liver abscess", "hepatic cyst", "hydatid cyst", "liver metastasis",
        
        # 特定并发症（肝病专属）
        "hepatic encephalopathy", "hepatorenal syndrome", "HRS",
        "spontaneous bacterial peritonitis", "SBP", "variceal bleeding",
        "hepatopulmonary syndrome", "portopulmonary hypertension"
    ],
    
    # =============== 中置信度（需与解剖词共现）===============
    "medium_confidence": [
        # 症状/体征（防非肝源性误筛）
        "jaundice", "icterus", "hepatomegaly", "splenomegaly", "ascites",
        "spider angioma", "palmar erythema", "caput medusae", "asterixis",
        "hepatic flap", "pruritus"  # PBC典型症状
        
        # 诊断指标（需肝脏上下文）
        "ALT", "AST", "GGT", "ALP", "bilirubin", "total bilirubin", "direct bilirubin",
        "albumin", "INR", "PT", "prothrombin time", "platelet count", "thrombocytopenia",
        "APRI", "FIB-4", "ELF score", "enhanced liver fibrosis",
        "steatosis", "hepatocyte ballooning", "lobular inflammation", "fibrosis stage",
        
        # 检查/治疗（需肝脏上下文）
        "FibroScan", "liver stiffness", "CAP score", "transient elastography",
        "liver biopsy", "liver ultrasound", "CT liver", "MRI liver", "MRCP",
        "liver transplantation", "liver transplant", "TIPS", "porto-systemic shunt",
        "antiviral therapy", "direct-acting antivirals", "DAAs", "sofosbuvir",
        "ledipasvir", "resmetirom", "obeticholic acid", "ursodeoxycholic acid"
    ],
    
    # =============== 解剖上下文（验证中置信度词）===============
    "anatomy_context": [
        "liver", "hepatic", "hepatocyte", "hepatic parenchyma", "liver parenchyma",
        "portal vein", "hepatic vein", "hepatic artery", "liver lobule",
        "liver function", "liver disease", "liver disorder", "liver condition"
    ],
    
    # =============== 排除词（严格模式下辅助验证）===============
    "exclusion_phrases": [
        r"cholecystitis", r"choledocholithiasis", r"gallstone", 
        r"pancreatitis", r"pancreatic cancer", r"myocardial infarction",
        r"rhabdomyolysis", r"muscle injury", r"bone disease",
        r"Gilbert syndrome", r"hemolytic anemia", r"sickle cell"
    ]
}

# ====== 智能筛选函数（含防误筛机制） ======
def is_liver_related(text):
    """判断文本是否属于肝病主题（非简单关键词匹配）"""
    text_lower = text.lower()
    
    # 1. 优先匹配高置信度核心病名（避免"肝脾肿大"误筛）
    if any(kw in text_lower for kw in LIVER_KEYWORDS_CN["core_diseases"]):
        return True
    
    # 2. 匹配肝脏特异性检查+上下文验证（防"黄疸"泛化）
    if any(kw in text_lower for kw in LIVER_KEYWORDS_CN["liver_specific"]):
        # 排除典型非肝病场景（如新生儿溶血、胆道梗阻等）
        exclusion_phrases = ["新生儿溶血", "胆总管结石", "胰头癌", " Gilbert", "溶血"]
        if not any(ep in text_lower for ep in exclusion_phrases):
            return True
    
    # 3. 中医术语精准匹配（避免"肝"字泛化）
    if any(kw in text_lower for kw in LIVER_KEYWORDS_CN["tcm"]):
        return True
    
    # 4. 谨慎处理"黄疸/肝肿大"：需与肝病关键词共现
    if ("黄疸" in text or "肝肿大" in text) and any(
        kw in text_lower for kw in LIVER_KEYWORDS_CN["core_diseases"] + LIVER_KEYWORDS_CN["liver_specific"]
    ):
        return True
    
    return False

def filter_liver_dataset(input_path, output_path):
    # 读取数据
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filtered = []
    matched_keywords = {}  # 统计关键词分布
    
    for item in data:
        # 合并三字段文本（保留原始结构）
        full_text = " ".join([
            item.get("Question", ""),
            item.get("Complex_CoT", ""),
            item.get("Response", "")
        ])
        
        if is_liver_related(full_text):
            filtered.append(item)
            # 记录匹配到的关键词（用于质量分析）
            for category, kws in LIVER_KEYWORDS.items():
                for kw in kws:
                    if kw in full_text.lower():
                        matched_keywords[kw] = matched_keywords.get(kw, 0) + 1
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    
    # 输出统计报告
    print(f"✅ 筛选完成！原始数据: {len(data)} 条 | 肝病相关: {len(filtered)} 条 ({len(filtered)/len(data):.1%})")
    print(f"\n📊 高频关键词分布（前10）:")
    for kw, cnt in sorted(matched_keywords.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   • {kw}: {cnt} 次")
    
    # 示例展示（避免泄露敏感信息）
    if filtered:
        print(f"\n🔍 筛选示例（Question片段）:")
        for i, item in enumerate(filtered[:3], 1):
            q = item["Question"][:80] + "..." if len(item["Question"])>80 else item["Question"]
            print(f"{i}. {q}")
    
    return filtered


def is_liver_related_en(text, strict_mode=True):
    """肝病主题精准判断（三重验证机制）"""
    text_lower = text.lower()
    
    # 1️⃣ 高置信度词：直接保留（如"HCC", "cirrhosis"）
    for kw in LIVER_KEYWORDS_EN["high_confidence"]:
        if re.search(rf'\b{re.escape(kw.lower())}\b', text_lower):
            return True
    
    # 2️⃣ 中置信度词 + 解剖上下文：共现验证（防"jaundice due to gallstone"误筛）
    has_medium = any(
        re.search(rf'\b{re.escape(kw.lower())}\b', text_lower) 
        for kw in LIVER_KEYWORDS_EN["medium_confidence"]
    )
    has_context = any(
        re.search(rf'\b{re.escape(kw.lower())}\b', text_lower) 
        for kw in LIVER_KEYWORDS_EN["anatomy_context"]
    )
    if has_medium and has_context:
        # 严格模式：排除典型非肝病场景
        if strict_mode and any(
            re.search(phrase, text_lower) 
            for phrase in LIVER_KEYWORDS_EN["exclusion_phrases"]
        ):
            return False
        return True
    
    return False

def filter_liver_english(input_path, output_path):
    # 读取数据
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误：文件 '{input_path}' 不存在！")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ 错误：文件 '{input_path}' 不是有效的JSON格式！")
        sys.exit(1)
    
    # 筛选（严格检查三个字段）
    filtered = []
    for item in data:
        # 合并三个字段文本（保留原始结构）
        fields = ["Question", "Complex_CoT", "Response"]
        full_text = " ".join(str(item.get(f, "")) for f in fields)
        
        if is_liver_related_en(full_text):
            filtered.append(item)  # 完整保留原始字典结构
    
    # 保存结果（严格保持原格式）
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    
    # 输出专业报告
    print(f"✅ 筛选完成 | 原始: {len(data)} 条 | 肝病相关: {len(filtered)} 条 ({len(filtered)/len(data):.1%})")
    print(f"📁 输出文件: {output_path}")
    
    # 示例展示（脱敏）
    if filtered:
        print("\n🔍 筛选示例 (Question片段):")
        for i, item in enumerate(filtered[:3], 1):
            q = item["Question"][:120] + "..." if len(item["Question"]) > 120 else item["Question"]
            print(f"{i}. {q}")
    else:
        print("\n⚠️  警告：未筛选到任何肝病相关数据！请检查：")
        print("   • 数据集中是否包含肝病内容")
        print("   • 是否需要扩展关键词库（如添加'hepatomegaly', 'jaundice'等）")
    
    return filtered

# ====== 使用说明 ======
if __name__ == "__main__":
    # 修改为您的实际文件路径
    INPUT_FILE = "/jeyzhang/shenxiaofeng/Liver/HuatuoGPT-o1/data/medical_o1_sft.json"   # ← 替换为您的文件名
    OUTPUT_FILE = "./data/liver_disease_filtered_en.json"
    
    filter_liver_english(INPUT_FILE, OUTPUT_FILE)
    