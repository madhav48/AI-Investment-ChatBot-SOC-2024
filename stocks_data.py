
"""

This file contains all the stock related data required for fetching data..


"""


sectors = [
    "finance", "technology", "healthcare", "education", "energy", 
    "telecommunications", "manufacturing", "retail", "real estate", 
    "automotive", "transportation", "food and beverage", "entertainment", 
    "hospitality", "insurance", "pharmaceuticals", "biotechnology", 
    "aerospace", "defense", "construction", "agriculture", "environment", 
    "media", "advertising", "legal", "government", "non-profit", 
    "e-commerce", "startups", "software", "hardware", "consulting", 
    "financial services", "venture capital", "private equity", 
    "education technology", "renewable energy", "oil and gas", "mining", 
    "transport logistics", "cybersecurity", "artificial intelligence", 
    "machine learning", "robotics", "big data", "cloud computing", 
    "virtual reality", "augmented reality", "3D printing", "internet of things"
    ]


sector_stock_dict = {
    "technology": ["AAPL", "MSFT", "GOOGL", "META", "AMD", "INTC", "NVDA", "CSCO", "IBM", "ORCL"],
    "healthcare": ["JNJ", "PFE", "MRNA", "ABBV", "GILD", "NVAX", "BMY", "AMGN", "VRTX", "REGN"],
    "finance": ["JPM", "GS", "BAC", "C", "WFC", "MS", "USB", "BK", "SCHW", "BLK"],
    "education": ["EDU", "GHC", "APOL", "COCO", "LAUR"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "HAL", "VLO", "PSX", "MPC"],
    "telecommunications": ["VZ", "T", "TMUS", "CTL", "LUMN"],
    "manufacturing": ["CAT", "DE", "HON", "3M", "GE", "ROK", "EMR", "IR", "NUE", "X"],
    "retail": ["AMZN", "WMT", "TGT", "COST", "HD", "LOW", "M", "KSS", "TJX", "ROST"],
    "real estate": ["SPG", "O", "AVB", "PLD", "DRE", "WELL", "EQR", "PSA", "ESS", "HST"],
    "automotive": ["TSLA", "GM", "F", "NIO", "BYDDF", "XPEV", "RIVN", "LCID", "STLA", "HMC"],
    "transportation": ["UAL", "DAL", "LUV", "CSX", "NSC", "UPS", "FDX", "KEX", "GWW", "UNP"],
    "food and beverage": ["KO", "PEP", "MDLZ", "GIS", "CAG", "K", "SJM", "HSY", "TSN", "ADM"],
    "entertainment": ["DIS", "NFLX", "CMCSA", "TWTR", "ATVI", "EA", "ROKU", "FUBO", "VIAC", "SONY"],
    "hospitality": ["MAR", "HOT", "HLT", "IHG", "WYNN", "MGM", "LVS", "RCL", "NCLH", "CHH"],
    "insurance": ["BRK.B", "AIG", "PGR", "TRV", "MET", "PRU", "CINF", "HIG", "UNM", "AFL"],
    "pharmaceuticals": ["ABBV", "PFE", "MRNA", "GILD", "BMY", "AMGN", "REGN", "VRTX", "ISRG", "NVS"],
    "biotechnology": ["BIIB", "GILD", "VRTX", "AMGN", "REGN", "MRNA", "NVAX", "ALNY", "EDIT", "CRSP"],
    "aerospace": ["BA", "LMT", "NOC", "RTX", "GD", "HII", "AIR", "HEI", "ROK", "BALL"],
    "defense": ["LMT", "NOC", "GD", "BA", "RTX", "HII", "MANT", "CSLT", "PLTR", "LHX"],
    "construction": ["LEN", "DHI", "PHM", "NVR", "TOL", "HOV", "KBH", "MTH", "RYL", "BZH"],
    "agriculture": ["MON", "ADM", "CORN", "CF", "SYT", "NTR", "AGRO", "TSN", "FMC", "POT"],
    "environment": ["NTR", "SYT", "CZR", "VLO", "BEP", "NEE", "EIX", "BLX", "CENX", "CREE"],
    "media": ["DIS", "VIAC", "T", "CMCSA", "DISH", "SIRI", "AMC", "FUBO", "Roku", "LUMN"],
    "advertising": ["GOOGL", "META", "TWTR", "WPP", "OMC", "IPG", "PUBGY", "STO", "TBWA", "HUM"],
    "legal": ["DLA", "BCLP", "LIT", "HIL", "BGR", "MCC", "FO", "BL", "GEO", "HRG"],
    "government": ["BA", "NOC", "LMT", "SAIC", "DLA", "HII", "ALB", "NVT", "TRN", "MOG.A"],
    "non-profit": ["WWF", "UNICEF", "NGO", "CRS", "ACF", "HNF", "MED", "NGP", "HEAL", "GBL"],
    "e-commerce": ["AMZN", "EBAY", "SHOP", "PDD", "BABA", "ETSY", "JD", "RBLX", "FVRR", "WISH"],
    "startups": ["SPCE", "RBLX", "PSTG", "PLTR", "FVRR", "TRX", "MNDY", "SDGR", "AFRM", "SQ"],
    "software": ["MSFT", "ADBE", "CRM", "SAP", "ORCL", "INTU", "IBM", "NOW", "VEEV", "SQ"],
    "hardware": ["AAPL", "DELL", "HPE", "TSMC", "AMAT", "NVDA", "INTC", "WDC", "MU", "NXP"],
    "consulting": ["ACN", "Deloitte", "TTEC", "CG", "Bain", "MBB", "ATK", "PWC", "EY", "KPMG"],
    "financial services": ["JPM", "GS", "MS", "C", "BAC", "BLK", "SCHW", "TROW", "STT", "AON"],
    "investment": ["BRK.B", "VTI", "IVV", "SPY", "BND", "XLB", "XLC", "XLI", "XLF", "XLY"],
    "venture capital": ["NEA", "SEQUOIA", "ACCEL", "KPCB", "Bessemer", "Benchmark", "Greylock", "Union", "IVP", "GV"],
    "private equity": ["KKR", "TPG", "BXP", "CG", "BX", "AON", "CQR", "TPH", "OHI", "APO"],
    "education technology": ["EDU", "COE", "APOL", "GHC", "PRDO", "FAT", "TAL", "HHS", "WIL", "RLD"],
    "renewable energy": ["NEX", "PLUG", "FSLR", "SOL", "ENPH", "SPWR", "JKS", "SEDG", "TAN", "BE"],
    "oil and gas": ["XOM", "CVX", "COP", "EOG", "OXY", "HAL", "PXD", "SLB", "NOV", "RIG"],
    "mining": ["BHP", "RIO", "VALE", "FCX", "NEM", "GOLD", "TECK", "DRD", "SAND", "MGB"],
    "transport logistics": ["UPS", "FDX", "CHRW", "XPO", "JBHT", "ODFL", "R", "ABFS", "SAIA", "CNI"],
    "cybersecurity": ["PANW", "CRWD", "ZS", "S", "FTNT", "OKTA", "CYBR", "TENB", "NICE", "NLOK"],
    "artificial intelligence": ["GOOGL", "MSFT", "NVDA", "IBM", "AMZN", "META", "BIDU", "TSLA", "INTC", "COG"],
    "machine learning": ["GOOGL", "MSFT", "AMZN", "NVIDIA", "IBM", "AZPN", "BIDU", "SAP", "VMW", "GOOG"],
    "robotics": ["ROK", "ABB", "NVT", "IRBT", "KUKA", "FANUC", "YASKAWA", "OMRON", "TSLA", "AUB"],
    "big data": ["IBM", "GOOGL", "MSFT", "AMZN", "HDP", "SPLK", "TIBX", "CLDR", "DADA", "NET"],
    "cloud computing": ["AMZN", "MSFT", "GOOGL", "IBM", "CRM", "ADBE", "ORCL", "NOW", "VMW", "OKTA"],
    "virtual reality": ["Oculus", "HTC", "VRX", "GOOGL", "MSFT", "META", "AMD", "NVIDIA", "ARVR", "SENS"],
    "augmented reality": ["AAPL", "GOOGL", "MSFT", "META", "SNAP", "NI", "Vuzix", "DAI", "BIDU", "TTWO"],
    "3D printing": ["DDD", "SSYS", "XONE", "HPQ", "PRLB", "GE", "MTU", "AMZN", "TTC", "NNDM"],
    "internet of things": ["GOOGL", "MSFT", "AMZN", "TSLA", "AAPL", "IBM", "NXP", "QCOM", "AVGO", "SENSOR"]
}
